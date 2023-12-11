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
                 thresh_value: float = None, id: str = ""):
        """
        To create a neuron, you must give it input(s) with weights for each input. If the neuron has other neurons as
        input, the 'parent' neuron so to speak will be calculated automatically.
        :param inputs: The structure of an input is (input_value, input_weight).
        :param thresh_value: the value at which the unit will fire.
        :param id: The identifier of a specific neuron (helps with debugging).
        """
        self.inputs: List[Union[Tuple[Neuron, float], Tuple[float, float]]] = inputs
        self.id: str = id
        self.input_weights: List[float] = [_[1] for i, _ in enumerate(self.inputs)]
        self.bias: float = -thresh_value
        self.activation_value = None
        self.cost_value = None
        self.error_delta = None

    def delta_rule(self, error_deltas=None, weights=None):
        output_sum: float = 0
        for index in range(0, len(error_deltas)):
            output_sum += error_deltas[index] * weights[index]
        self.delta = (1 - math.tanh(self.activation_value) ** 2) * output_sum
        return self.delta, self.input_weights

    def calc_delta_rule(self, desired_outcome: int):
        self.error_delta = (1 - math.tanh(self.activation_value) ** 2) * (desired_outcome - self.activation_value)
        return self.error_delta, self.input_weights

    def update(self, lr: float):
        for index, (neuron, weight) in enumerate(self.inputs):
            if neuron.get_activation_value() is not None:
                self.inputs[index] = (neuron, lr * self.error_delta * neuron.get_activation_value())
        self.bias += lr * self.error_delta

    def sigmoid(self, activation_value):
        # return result of sigmoid function
        return 1.0 / (1.0 + exp(-activation_value))

    def sigmoid_derivative(self, sigmoid_outcome: float):
        return sigmoid_outcome * (1.0 - sigmoid_outcome)

    def activate(self, weights: List[float], inputs: List[float]):
        activation: float = sum(np.array(weights) * np.array(inputs)) + self.bias
        return activation

    def calc_activation(self) -> float:
        """
        This recursive function will calculate the weighted sum of inputs.
        :param invalidate_stored_input: set to True to recalculate all outputs (used for weight updates).
        :return: 1 or 0.
        """
        input_weights: List[float] = []
        activations: List[float] = []
        input_neuron: Union[Tuple[Neuron, float], Tuple[float, float]]
        value, weight = self.inputs[0]

        # arrived at input neuron (ie root-node)
        # plainly perform calculation with input value
        if isinstance(value, float):
            # add result to sum of weighted inputs of this neuron
            input_weights.append(weight)
            activations.append(value * weight)

        else:  # value is a Neuron
            # invalidate everything when the root (input neurons) has been changed!
            for neuron, weight in self.inputs:
                # calculate output of the other neuron
                neuron_result: float = neuron.get_activation_value() * weight
                # add the output of the other node to the sum of the weighted inputs of this neuron
                input_weights.append(weight)
                activations.append(neuron_result)

        return self.sigmoid(activation_value=self.activate(input_weights, activations))

    def get_activation_value(self):
        """
        This funcion works as a cache manager
        :return:
        """
        if self.activation_value is None:
            self.activation_value = self.calc_activation()
        return self.activation_value

    def clear(self):
        self.activation_value = None

    def put(self, input: float):
        """
        This function invalidates old outputs.
        Note that the weight of this input is already stored in the neuron, so we shouldn't touch it.
        :param input: New value to use as input for an input neuron
        :return: None
        """
        self.inputs = [(input, self.inputs[0][1])]
        self.activation_value = None
        return

    def __repr__(self):
        return f"{self.id} bias:{self.bias} w:{self.input_weights} act:{self.activation_value} "


class NeuralNet:
    """
    Todo: document class
    """

    def __init__(self, test_set: List[List[float]], validation_value: List[List[float]], n_neurons_per_layer: int = 4,
                 n_hidden_layers: int = 6, n_outputs: int = 3):
        """
        Todo: document init func
        :param test_set:
        :param validation_value:
        :param n_neurons_per_layer:
        :param n_hidden_layers:
        """
        self.output_deltas = []
        self.output_weights = []
        self.test_set: List[List[float]] = test_set
        self.validate_set: int = validation_value
        self.n_neurons_per_layer: int = n_neurons_per_layer
        self.n_hidden_layers: int = n_hidden_layers

        self.input_layer: List[Neuron] = []
        self.hidden_layers: List[List[Neuron]] = []
        self.output_layer: List[Neuron] = []
        self.all_layers: List[List[Neuron]] = []
        self.all_neurons: List[Neuron] = []

        # account for the validation value at the end, so len - 1
        n_inputs = len(test_set[0]) - 1
        for _ in range(n_inputs):
            # initialize inputs with random thresholds
            new_neuron = Neuron([(0, random.random())], random.random(), f"input X{_}")
            self.input_layer.append(new_neuron)
            self.all_neurons.append(new_neuron)
        self.all_layers.append(self.input_layer)

        for _ in range(n_hidden_layers):
            layer: List[Neuron] = []
            for __ in range(n_neurons_per_layer):
                # first hidden layer needs to use the input neurons
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
        for _ in range(n_outputs):
            new_neuron = Neuron([(prev_neuron, random.random()) for prev_neuron in self.hidden_layers[-1]],
                                random.random(), f"output X{_}")
            self.output_layer.append(new_neuron)
            self.all_neurons.append(new_neuron)
        self.all_layers.append(self.output_layer)

    def clear_neuron_cache(self):
        map(lambda neuron: neuron.clear(), self.all_neurons)

    def forward_propagate(self, desired_output):

        # setting the desired neuron outcomes for the given label
        if desired_output == 0:
            desired_output = [1, 0, 0]
        elif desired_output == 1:
            desired_output = [0, 1, 0]
        else:  # y_true == 2
            desired_output = [0, 0, 1]

        neuron: Neuron
        # output layer
        for neuron_index, neuron in enumerate(self.all_layers[-1]):
            neuron.get_activation_value()
            error_delta, weights = neuron.calc_delta_rule(desired_output[neuron_index])
            self.output_deltas.append(error_delta)
            self.output_weights.append(weights)
        self.output_weights = np.array(self.output_weights).transpose().tolist()

        for layer_index in range(len(self.all_layers) - 2, 0, -1):
            prev_output_deltas = self.output_deltas
            prev_output_weights = self.output_weights
            self.output_deltas = []
            self.output_weights = []
            for neuron_index, neuron in enumerate(self.all_layers[layer_index]):
                error_delta, weights = neuron.delta_rule(prev_output_deltas, prev_output_weights[neuron_index])
                self.output_deltas.append(error_delta)
                self.output_weights.append(weights)

            self.output_weights = np.array(self.output_weights).transpose().tolist()


    def back_propagate(self, lr: float = 0.1) -> None:
        """
        Update all neurons starting at the output going backwards.
        :return: None
        """
        neuron: Neuron
        [neuron.update(lr) for _, neuron in enumerate(reversed(self.all_neurons))]
        # map(lambda neuron: neuron.update(), self.output_layer)
        # map(lambda layer: map(lambda neuron: neuron.update(), layer), reversed(self.hidden_layers))
        # map(lambda neuron: neuron.update(), self.input_layer)

    def generate_output(self):
        self.clear_neuron_cache()
        return [neuron.get_activation_value() for _, neuron in enumerate(self.output_layer)]


def convert(datarow: int):
    # setting the desired neuron outcomes for the given label
    if datarow == 0:
        desired_output = [1, 0, 0]
    elif datarow == 1:
        desired_output = [0, 1, 0]
    else:  # datarow[-1] == 2
        desired_output = [0, 0, 1]
    return desired_output

