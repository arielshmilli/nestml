# -*- coding: utf-8 -*-
#
# co_co_convolve_cond_correctly_built.py
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

from typing import Any, Dict, Optional

try:
    # Available in the standard library starting with Python 3.12
    from typing import override
except ImportError:
    # Fallback for Python 3.8 - 3.11
    from typing_extensions import override

from pynestml.cocos.co_co import CoCo
from pynestml.meta_model.ast_model import ASTModel
from pynestml.meta_model.ast_node import ASTNode
from pynestml.meta_model.ast_simple_expression import ASTSimpleExpression
from pynestml.symbols.symbol import SymbolKind
from pynestml.symbols.unit_type_symbol import UnitTypeSymbol
from pynestml.symbols.variable_symbol import BlockType
from pynestml.utils.ast_utils import ASTUtils
from pynestml.utils.logger import LoggingLevel, Logger
from pynestml.utils.messages import Messages
from pynestml.visitors.ast_visitor import ASTVisitor


class CoCoConvolveCondCorrectlyBuilt(CoCo):
    """
    This coco ensures that ``convolve`` is correctly called, i.e. that the first argument is the variable from the state block and the second argument is a spiking input port.

    Allowed:
        inline I_syn_exc pA = convolve(g_exc, exc_spikes) * ( V_m - E_exc )

    Not allowed:
        inline I_syn_exc pA = convolve(g_exc, g_exc) * ( V_m - E_exc )
        inline I_syn_exc pA = convolve(exc_spikes, g_exc) * ( V_m - E_exc )
    """

    @classmethod
    @override
    def check_co_co(cls, node: ASTNode, metadata: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Ensures the coco for the handed over neuron.
        :param node: a single model instance.
        """
        assert isinstance(node, ASTModel), "This coco can only be called on ASTModels!"

        node.accept(ConvolveCheckerVisitor(node))


class ConvolveCheckerVisitor(ASTVisitor):
    """
    Visits a function call and checks that if the function call is a convolve, the parameters are correct.
    """

    def __init__(self, model: ASTModel):
        super().__init__()
        self._model = model

    def visit_function_call(self, node):
        func_name = node.get_name()
        if func_name == "convolve":
            symbol_var = node.get_scope().resolve_to_symbol(str(node.get_args()[0]),
                                                            SymbolKind.VARIABLE)
            port_arg = node.get_args()[1]
            port_name = port_arg.get_variable().get_complete_name() if isinstance(port_arg, ASTSimpleExpression) and port_arg.is_variable() else str(port_arg)
            symbol_port = node.get_scope().resolve_to_symbol(port_name,
                                                             SymbolKind.VARIABLE)
            if symbol_var is not None and not symbol_var.is_kernel() and not symbol_var.is_state():
                code, message = Messages.get_first_arg_not_kernel_or_equation(func_name)
                Logger.log_message(code=code, message=message,
                                   error_position=node.get_source_position(), log_level=LoggingLevel.ERROR)
            if symbol_port is not None and not symbol_port.is_spike_input_port():
                code, message = Messages.get_second_arg_not_a_spike_port(func_name)
                Logger.log_message(error_position=node.get_source_position(),
                                   code=code, message=message,
                                   log_level=LoggingLevel.ERROR)

            if isinstance(port_arg, ASTSimpleExpression) and port_arg.is_variable() and port_arg.get_variable().is_delay_variable():
                self._check_delayed_port(node, port_arg.get_variable())

    def _check_delayed_port(self, node, port_var) -> None:
        port_name = port_var.get_name()
        delay_parameter = port_var.get_delay_parameter()

        def error(reason: str) -> None:
            code, message = Messages.get_delayed_convolution_invalid(port_name, delay_parameter, reason)
            Logger.log_message(error_position=node.get_source_position(), code=code, message=message, log_level=LoggingLevel.ERROR)

        delay_symbol = node.get_scope().resolve_to_symbol(delay_parameter, SymbolKind.VARIABLE)
        if delay_symbol is None or delay_symbol.block_type != BlockType.PARAMETERS:
            error("the delay has to be a parameter")
        elif not (isinstance(delay_symbol.get_type_symbol(), UnitTypeSymbol) and delay_symbol.get_type_symbol().unit.unit.physical_type == "time"):
            error("the delay has to have a unit of time")

        if delay_parameter.startswith("_") or "__" in delay_parameter:
            error("the name of the delay parameter cannot start with an underscore or contain a double underscore")

        if port_var.has_vector_parameter():
            error("vectors of spiking input ports cannot be delayed")

        kernel_arg = node.get_args()[0]
        if isinstance(kernel_arg, ASTSimpleExpression) and kernel_arg.is_variable():
            kernel = self._model.get_kernel_by_name(kernel_arg.get_variable().get_name())
            if kernel is not None and ASTUtils.is_delta_kernel(kernel):
                error("delta kernels cannot be delayed; add the delay to the connection instead")
