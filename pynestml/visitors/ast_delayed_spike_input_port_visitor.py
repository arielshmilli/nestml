# -*- coding: utf-8 -*-
#
# ast_delayed_spike_input_port_visitor.py
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

from typing import Optional

from pynestml.meta_model.ast_expression import ASTExpression
from pynestml.meta_model.ast_function_call import ASTFunctionCall
from pynestml.meta_model.ast_node_factory import ASTNodeFactory
from pynestml.meta_model.ast_simple_expression import ASTSimpleExpression
from pynestml.symbols.predefined_functions import PredefinedFunctions
from pynestml.symbols.predefined_variables import PredefinedVariables
from pynestml.visitors.ast_visitor import ASTVisitor


class ASTDelayedSpikeInputPortVisitor(ASTVisitor):
    r"""
    Converts a delayed spike input port in the second argument of a convolution, ``convolve(K, spikes(t - delay))``, into the variable ``spikes`` with ``delay`` as its delay parameter.

    The parser reads ``spikes(t - delay)`` as a function call. The AST builder converts it as soon as the convolve call is built (see ``ASTBuilderVisitor.visitFunctionCall()``), so that the checks and type calculations that follow see an ordinary spike input port, also in expressions that are parsed again from strings later on. Function calls that do not have the form ``name(t - variable)`` are left alone; the convolve checks report them.
    """

    def visit_function_call(self, node: ASTFunctionCall) -> None:
        self.convert_delayed_port(node)

    @classmethod
    def convert_delayed_port(cls, node: ASTFunctionCall) -> None:
        r"""
        If ``node`` is a convolution with a delayed spike input port, replace the port's function call by a variable with a delay parameter.
        """
        if node.get_name() != PredefinedFunctions.CONVOLVE or len(node.get_args()) != 2:
            return

        port_arg = node.get_args()[1]
        if not isinstance(port_arg, ASTSimpleExpression) or not port_arg.is_function_call():
            return

        delayed_port = port_arg.get_function_call()
        delay_parameter = cls._get_delay_parameter(delayed_port)
        if delay_parameter is None:
            return

        port_var = ASTNodeFactory.create_ast_variable(delayed_port.get_name(), source_position=delayed_port.get_source_position())
        port_var.set_delay_parameter(delay_parameter)
        port_arg.set_function_call(None)
        port_arg.set_variable(port_var)

    @staticmethod
    def _get_delay_parameter(delayed_port: ASTFunctionCall) -> Optional[str]:
        r"""
        Return the name of ``delay`` if the function call has the form ``name(t - delay)``, else None.
        """
        if len(delayed_port.get_args()) != 1:
            return None

        arg = delayed_port.get_args()[0]
        if not isinstance(arg, ASTExpression) or not getattr(arg.get_binary_operator(), "is_minus_op", False):
            return None

        lhs, rhs = arg.get_lhs(), arg.get_rhs()
        if not (isinstance(lhs, ASTSimpleExpression) and lhs.is_variable() and lhs.get_variable().get_name() == PredefinedVariables.TIME_CONSTANT):
            return None

        if not (isinstance(rhs, ASTSimpleExpression) and rhs.is_variable()):
            return None

        return rhs.get_variable().get_name()
