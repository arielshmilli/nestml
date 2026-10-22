# -*- coding: utf-8 -*-
#
# test_delayed_convolutions.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import os
import pytest

import nest

from pynestml.codegeneration.nest_tools import NESTTools
from pynestml.frontend.pynestml_frontend import generate_nest_target


@pytest.mark.skipif(NESTTools.detect_nest_version().startswith("v2"),
                    reason="This test does not support NEST 2")
class TestDelayedConvolutions:
    r"""
    Test convolutions with a delayed spike input port, ``convolve(K, spikes(t - d))``, in neurons and synapses.
    """

    module_name = "nestml_delayed_convolutions_module"
    resolution = .1    # [ms]

    @pytest.fixture(scope="class", autouse=True)
    def setUp(self):
        resources = os.path.realpath(os.path.join(os.path.dirname(__file__), "resources"))
        generate_nest_target(input_path=[os.path.join(resources, "delayed_convolution_neuron.nestml"),
                                         os.path.join(resources, "delayed_convolution_relay_neuron.nestml"),
                                         os.path.join(resources, "delayed_convolution_synapse.nestml")],
                             logging_level="DEBUG",
                             module_name=self.module_name,
                             suffix="_nestml",
                             codegen_opts={"neuron_synapse_pairs": [{"neuron": "delayed_convolution_relay_neuron",
                                                                     "synapses": {"delayed_convolution_synapse": {"post_ports": ["post_spikes"]}}}],
                                           "weight_variable": {"delayed_convolution_synapse": "w"},
                                           # these change only on postsynaptic spikes, and would otherwise be moved to the neuron, together with the convolutions they read
                                           "strictly_synaptic_vars": {"delayed_convolution_synapse": ["c_sum_at_post", "c2_sum_at_post"]}})

    def _reset(self):
        nest.ResetKernel()
        nest.SetKernelStatus({"resolution": self.resolution})
        try:
            nest.Install(self.module_name)
        except Exception:
            # ResetKernel() does not unload modules for NEST Simulator < v3.7; ignore exception if module is already loaded on earlier versions
            pass

    @pytest.mark.parametrize("neuron_model_name", ["delayed_convolution_neuron_nestml", "delayed_convolution_nonlinear_neuron_nestml"])
    @pytest.mark.parametrize("kernel_delay", [0., 3., 5.])
    def test_delayed_convolution_neuron(self, neuron_model_name: str, kernel_delay: float):
        r"""
        The neuron convolves the kernel with the undelayed spikes and, with twice the weight, with the spikes delayed by ``kernel_delay``.
        The reference neuron gets the same input through two connections whose delays differ by ``kernel_delay``.
        """
        reference_model_name = neuron_model_name.replace("_neuron_nestml", "_reference_neuron_nestml")
        connection_delay = 1.    # [ms]
        weight = 100.
        spike_times = [2., 4., 4.5, 10., 10.1, 30., 31.3, 33., 60.]

        self._reset()
        sg = nest.Create("spike_generator", params={"spike_times": spike_times})
        neuron = nest.Create(neuron_model_name, params={"kernel_delay": kernel_delay})
        reference = nest.Create(reference_model_name)
        nest.Connect(sg, neuron, syn_spec={"weight": weight, "delay": connection_delay})
        nest.Connect(sg, reference, syn_spec={"weight": weight, "delay": connection_delay})
        nest.Connect(sg, reference, syn_spec={"weight": 2 * weight, "delay": connection_delay + kernel_delay})

        mm_neuron = nest.Create("multimeter", params={"record_from": ["V_m", "I_syn"], "interval": self.resolution})
        mm_reference = nest.Create("multimeter", params={"record_from": ["V_m", "I_syn"], "interval": self.resolution})
        nest.Connect(mm_neuron, neuron)
        nest.Connect(mm_reference, reference)

        # two calls, so that spikes are still in transit in the delay buffer between them
        nest.Simulate(31.)
        nest.Simulate(69.)

        for var in ["V_m", "I_syn"]:
            x = mm_neuron.get("events")[var]
            x_ref = mm_reference.get("events")[var]
            assert np.amax(np.abs(x_ref)) > 0.
            np.testing.assert_allclose(x, x_ref, rtol=1E-9, atol=1E-12)

    def test_delayed_convolution_neuron_change_delay(self):
        r"""The delay can be changed between simulations, but not while spikes are in transit."""
        self._reset()
        sg = nest.Create("spike_generator", params={"spike_times": [18.]})
        neuron = nest.Create("delayed_convolution_neuron_nestml")
        nest.Connect(sg, neuron, syn_spec={"weight": 100., "delay": 1.})

        nest.Simulate(10.)
        neuron.kernel_delay = 3.    # nothing in transit yet
        nest.Simulate(10.)

        neuron.kernel_delay = 4.    # the spike at 18 ms arrives at 19 ms and is delayed until 22 ms
        with pytest.raises(Exception, match="cannot be changed while spikes are in transit"):
            nest.Simulate(10.)

    @staticmethod
    def _trace(t: float, release_times: np.ndarray, tau: float) -> float:
        r"""value at time ``t`` of the sum of exponential traces that start at ``release_times``; a release at exactly ``t`` counts"""
        dt = t - release_times[release_times <= t]
        return np.sum(np.exp(-dt / tau))

    @staticmethod
    def _trace_integral(t: float, release_times: np.ndarray, tau: float) -> float:
        r"""integral from 0 to ``t`` of the sum of exponential traces that start at ``release_times``"""
        dt = t - release_times[release_times < t]
        return np.sum(tau * (1. - np.exp(-dt / tau)))

    def test_delayed_convolution_synapse(self):
        r"""
        Compare the traces that the synapse sees at pre- and postsynaptic spikes, and their integral over time, with the analytic values.
        The presynaptic spikes include a burst of 20 spikes within 19 ms, more than the initial capacity of the release queue.
        """
        neuron_model_name = "delayed_convolution_relay_neuron_nestml__with_delayed_convolution_synapse_nestml"
        synapse_model_name = "delayed_convolution_synapse_nestml__with_delayed_convolution_relay_neuron_nestml"
        tau_c = 48.8    # [ms]
        delays = {"c": 18.8, "c2": 7.3}    # [ms]
        syn_delay = 1.    # [ms]
        pre_times = np.concatenate([[10., 15., 22., 50., 51.], 120. + np.arange(20), [181.7, 230., 300.]])
        post_drive_times = [5.5, 12.3, 27.4, 40.6, 63.1, 125.5, 137.6, 170.1, 205.2, 260.8]

        self._reset()
        sg_pre = nest.Create("spike_generator", params={"spike_times": pre_times})
        sg_post = nest.Create("spike_generator", params={"spike_times": post_drive_times})
        pre_neuron = nest.Create("parrot_neuron")
        post_neuron = nest.Create(neuron_model_name)
        sr_pre = nest.Create("spike_recorder")
        sr_post = nest.Create("spike_recorder")

        nest.Connect(sg_pre, pre_neuron)
        nest.Connect(sg_post, post_neuron, syn_spec={"weight": 1.})
        nest.Connect(pre_neuron, post_neuron, syn_spec={"synapse_model": synapse_model_name, "weight": 0., "delay": syn_delay})
        nest.Connect(pre_neuron, sr_pre)
        nest.Connect(post_neuron, sr_post)

        # two calls, so that releases are still pending in the synapse between them
        nest.Simulate(125.)
        nest.Simulate(200.)

        t_pre = np.array(sr_pre.get("events")["times"])
        t_post = np.array(sr_post.get("events")["times"]) + syn_delay    # postsynaptic spikes reach the synapse after the dendritic delay
        assert len(t_pre) == len(pre_times)
        assert len(t_post) == len(post_drive_times)
        t_post = t_post[t_post < t_pre[-1]]    # the synapse processes postsynaptic spikes only up to its last presynaptic spike

        syn = nest.GetConnections(source=pre_neuron, synapse_model=synapse_model_name)
        for var, delay in delays.items():
            release_times = t_pre + delay
            assert not np.any(np.isin(np.round(release_times, 6), np.round(np.concatenate([t_pre, t_post]), 6))), "test spike times should not coincide with release times"

            expected_sum_at_pre = sum(self._trace(t, release_times, tau_c) for t in t_pre)
            expected_sum_at_post = sum(self._trace(t, release_times, tau_c) for t in t_post)
            expected_integral = self._trace_integral(t_pre[-1], release_times, tau_c)

            np.testing.assert_allclose(syn.get(var + "_sum_at_pre"), expected_sum_at_pre, rtol=1E-10)
            np.testing.assert_allclose(syn.get(var + "_sum_at_post"), expected_sum_at_post, rtol=1E-10)
            np.testing.assert_allclose(syn.get(var + "_integral"), expected_integral, rtol=1E-10)
