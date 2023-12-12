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

import math
import random
from math import exp
from typing import List, Tuple, Union

import numpy as np


class Neuron:
    """
    This is an artificial neuron (unit).
    """

    def __init__(self, inputs: List[Union[Tuple[Neuron, float], Tuple[float, float]]] = None,
                 thresh_value: float = None, id: str = "") -> None:
        """
        To create a neuron, you must give it input(s) with weights for each input. If the neuron has other neurons as
        input, the 'parent' neuron so to speak will be calculated automatically.
        :param inputs: The structure of an input is (input_value, input_weight).
        :param thresh_value: the value at which the unit will fire.
        :param id: The identifier of a specific neuron (helps with debugging).
        """
        inputs: List[Union[Tuple[Neuron, float], Tuple[float, float]]]
        self.inputs: List[Neuron] = [i[0] for _, i in enumerate(inputs)]
        self.input_weights: List[float] = [j for _, (i, j) in enumerate(inputs)]
        self.id: str = id
        self.bias: float = -thresh_value

        # values needed for calculations & learning ability
        self.delta = None
        self.output_value = None
        self.z = None
        self.cost_value = None
        self.error_delta = None

    def delta_rule(self, error_deltas=None, weights=None) -> Tuple[float, List[float]]:
        """
        This function is used to calculate the delta value for a hidden layer neuron. It takes in the deltas of the
        neurons in the next layer and the weights of the connections between the neurons in the next layer
        and the current neuron. It then calculates the delta value for the current neuron and returns it.

        :param error_deltas: A list of the deltas of the neurons in the next layer.
        :param weights: A list of the weights of the connections between the neurons in the next layer and the current neuron.
        :return delta: The delta value for the current neuron.
        :return input_weights: A list of the weights of the connections between the neurons in the previous layer and the current neuron.
        """
        output_sum: float = 0
        for index in range(0, len(error_deltas)):
            output_sum += error_deltas[index] * weights[index]
        self.delta = (1 - math.tanh(self.z) ** 2) * output_sum
        return self.delta, self.input_weights

    def calc_delta_value(self, desired_outcome: List[int]) -> Union[float, List[float]]:
        """
        This function calculates the delta value for a neuron. The delta is the tanh derivative multiplied by a desired outcome parameter.
        @param desired_outcome: A list of numbers that indicate the desired outcome per output neuron.
        @return: the calculated error_delta and the list of input_weights
        """
        self.error_delta = (1 - math.tanh(self.z) ** 2) * (desired_outcome - self.output_value)
        return self.error_delta, self.input_weights

    def update(self, lr: float) -> None:
        """
        Updates the weights and bias of a neuron.
        @param lr: the learning rate (float) is used to slow down, or speed up learning.
        @return: None
        """
        for index, neuron in enumerate(self.inputs):
            self.input_weights[index] += lr * self.error_delta * neuron.output_value
        self.bias += lr * self.error_delta

    def calculate_output(self) -> float:
        """
        This function calculates the output of a neuron (activation value). It does this by combining the weighted values of its inputs.
        @return: the neurons output value.
        """
        input_sum = 0
        if not isinstance(self.inputs[0], float):
            for index in range(0, len(self.inputs)):
                input_sum += self.inputs[index].calculate_output() * self.input_weights[index]
        else:
            input_sum = sum(self.inputs)
        self.z = self.bias + input_sum
        self.output_value = math.tanh(self.z)
        return self.output_value

    def clear(self) -> None:
        # stop traversing through neurons
        if isinstance(self.inputs[0], float):
            return None
        else:
            neuron: Neuron
            for neuron in self.inputs:
                neuron.output_value = neuron.clear()

    def put(self, input: float) -> None:
        """
        This function puts some data values in the first layer of the neural network,
        after which the old output values of the neuron are invalidated.
        Note that the weight of this input is already stored in the neuron, so this function shouldn't touch it.
        :param input: New value to use as input for an input neuron
        :return: None
        """
        self.inputs = [input]
        self.output_value = None

    def __repr__(self) -> str:
        """
        Simple visualization function to 'pretty' print a neuron while debugging.
        @return: textbased neuron internals.
        """
        return f"{self.id} b:{self.bias} w:{self.input_weights} a:{self.output_value}"


class NeuralNet:
    """
    Todo: document class
    """

    def __init__(self, train_set: List[List[float]],
                 network_shape: Tuple[int, int, int],
                 learning_rate: float) -> None:
        """
        Behold the monster function... This initializer seems daunting, but it only sets up initial neuron layers and values
        (See the inline comments beneath for more details).
        :param train_set: This is the training dataset (It contains the data class).
        :param network_shape: This is the general shape of the network. The tuple means:
        (n neurons per layer, n hidden layers, n output neurons)
        Sadly the network only accepts as many neurons per layer, as there are input features.
        The amount of hidden layers is adaptable though.
        :param learning_rate: This float is the learning rate used for the updating of the internal neurons.
        """
        # learning stuff
        self.output_deltas: List[float] = []
        self.output_weights: List[float] = []
        self.learning_rate = learning_rate

        # data stuff
        self.test_set: List[List[float]] = train_set

        # network shape
        self.n_neurons_per_layer: int = network_shape[0]
        self.n_hidden_layers: int = network_shape[1]
        self.n_outputs = network_shape[2]

        # neuron layer stuff (extra detailed for debugging / educational purposes)
        self.input_layer: List[Neuron] = []
        self.hidden_layers: List[List[Neuron]] = []
        self.output_layer: List[Neuron] = []
        self.all_layers: List[List[Neuron]] = []
        self.all_neurons: List[Neuron] = []

        # account for the validation value at the end, so actual amount of features is len - 1
        n_inputs = len(train_set[0]) - 1

        # setting up the input layer
        for _ in range(n_inputs):
            # initialize inputs with random thresholds
            new_neuron = Neuron([(0, random.random())], random.random(), f"input X{_}")
            self.input_layer.append(new_neuron)
            self.all_neurons.append(new_neuron)
        self.all_layers.append(self.input_layer)

        # setting up the hidden layers
        for _ in range(self.n_hidden_layers):
            layer: List[Neuron] = []
            for __ in range(self.n_neurons_per_layer):
                # first hidden layer needs to use the input neurons instead of hidden layer neurons.
                if _ == 0:
                    # generating the neurons for each layer
                    # (Note that each layer neuron takes every previous neuron as input).
                    # the threshold for every neuron is randomized.
                    new_neuron = Neuron([(input_neuron, random.random()) for input_neuron in self.input_layer],
                                        random.random(),
                                        f"L{_}:N{__}")
                    layer.append(new_neuron)
                    self.all_neurons.append(new_neuron)
                else:  # every other layer should take previous layer as inputs
                    new_neuron = Neuron([(prev_neuron, random.random()) for prev_neuron in self.hidden_layers[-1]],
                                        random.random(), f"L{_}:N{__}")
                    layer.append(new_neuron)
                    self.all_neurons.append(new_neuron)
            self.hidden_layers.append(layer)
            self.all_layers.append(layer)

        # setting the last neurons as output
        for _ in range(self.n_outputs):
            new_neuron = Neuron([(prev_neuron, random.random()) for prev_neuron in self.hidden_layers[-1]],
                                random.random(), f"output X{_}")
            self.output_layer.append(new_neuron)
            self.all_neurons.append(new_neuron)
        self.all_layers.append(self.output_layer)

    def forward_propagate(self, desired_output: int) -> None:
        """
        The forward propagation is performed on the network. This function calculates the delta values for every neuron
        using a desired outcome value. See the delta_rule and calc_delta_value documentations on the exact calculation.
        @param desired_output: integer that indicates the correct given data class.
        @return: None
        """

        neuron: Neuron
        # output layer
        for neuron_index, neuron in enumerate(self.all_layers[-1]):
            neuron.calculate_output()
            error_delta, weights = neuron.calc_delta_value(convert(desired_output)[neuron_index])
            self.output_deltas.append(error_delta)
            self.output_weights.append(weights)
        self.output_weights = np.array(self.output_weights).transpose().tolist()

        # middle layers, but stop before reaching the start
        for layer_index in range(len(self.all_layers) - 2, 0, -1):
            prev_output_deltas: List[float] = self.output_deltas
            prev_output_weights: List[float] = self.output_weights
            self.output_deltas = []
            self.output_weights = []
            for neuron_index, neuron in enumerate(self.all_layers[layer_index]):
                error_delta, weights = neuron.delta_rule(prev_output_deltas, prev_output_weights[neuron_index])
                self.output_deltas.append(error_delta)
                self.output_weights.append(weights)

            self.output_weights = np.array(self.output_weights).transpose().tolist()

    def back_propagate(self) -> None:
        """
        Update all neurons starting at the output going backwards.
        @param lr: the learning rate by which the neuron values will change
        @return: None
        """
        neuron: Neuron
        for neuron in self.output_layer:
            neuron.update(self.learning_rate)

    def generate_output(self) -> List[float]:
        """
        This function is mainly used to validate a network. Quickly calculate all output values for the output layer.
        @return: the output values of the last layer, Can be interpreted as the networks 'guess' as to
        which data class belongs to a given set of input features.
        """
        return [neuron.calculate_output() for _, neuron in enumerate(self.output_layer)]


def convert(dataclass: int) -> List[int]:
    """
    dirty and quick way to generate a list of desired outputs from a single class integer.
    @param dataclass: the dataclass belonging to a row of features.
    @return: the desired outputs pertaining to a neural network that has 3 output nodes.
    """
    # setting the desired neuron outcomes for the given label
    if dataclass == 0:
        desired_output = [1, 0, 0]
    elif dataclass == 1:
        desired_output = [0, 1, 0]
    else:  # datarow[-1] == 2
        desired_output = [0, 0, 1]
    return desired_output
