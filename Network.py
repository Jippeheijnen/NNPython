#
# Created by Jippe Heijnen on 03-12-2023.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import annotations

from typing import List, Tuple, Union


class Neuron:
    """
    This is an artificial neuron (unit).
    """

    def __init__(self, inputs: List[Union[Tuple[Neuron, float], Tuple[float, float]]] = None,
                 thresh_value: float = None):
        """
        To create a neuron, you must give it input(s) with weights for each input. If the neuron has other neurons as
        input, the 'parent' neuron so to speak will be calculated automatically.
        :param inputs: The structure of an input is (input_value, input_weight).
        :param thresh_value: the value at which the unit will fire.
        """
        self.inputs: List[Union[Tuple[Neuron, float], Tuple[float, float]]] = inputs
        self.thresh_value: float = thresh_value
        self.output_value = None

        if thresh_value is not None:
            self.bias = -thresh_value
        else:
            self.bias = None

        # check if the output can be calculated
        if len(inputs) != 0 and thresh_value is not None:
            self.calc_output()

    def calc_output(self) -> float:
        """
        This function will calculate the weighted sum of inputs.
        :return: 1 or 0.
        """
        outputs: List[float] = []
        input_neuron: Union[Tuple[Neuron, float], Tuple[float, float]]

        for input_neuron in self.inputs:

            if type(input_neuron[0]) == Neuron:
                input_neuron_result: float = input_neuron[0].calc_output()
                outputs.append(float(input_neuron_result) * input_neuron[1])
            else:
                input_neuron_output: float = input_neuron[0] * input_neuron[1]
                outputs.append(input_neuron_output)

        self.output_value = 1 if sum(outputs) >= self.thresh_value else 0

        return self.output_value
