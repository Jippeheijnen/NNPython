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
from itertools import chain
import random
from math import inf
from typing import List, Union, Tuple

from pandas import DataFrame
from ucimlrepo import fetch_ucirepo

from Network import NeuralNet, convert


def find_high_low_values(data) -> Tuple[List[float], List[float]]:
    """
    This function goes through the dataset, and
    @param data:
    @return:
    """
    # setting infinities as placeholders
    low_values = [inf] * 4
    high_values = [-inf] * 4
    for row in data:
        for index in range(0, len(row) - 1):
            if row[index] < low_values[index]:
                low_values[index] = row[index]
            elif row[index] > high_values[index]:
                high_values[index] = row[index]
    return low_values, high_values


def normalize(data, low_values, high_values) -> List[List[float]]:
    """
        This function normalises the data in the data set.

        :param data: A list object with all data points
        :param low_values: A list object with all minimum values of all datasets for normalisation
        :param high_values: A list object with all maximum values of all datasets for normalisation
        :return A list object with all normalised data points
    """
    for index, datapoints in enumerate(data):
        for index in range(len(datapoints) - 1):
            data[index][index] = (datapoints[index] - low_values[index]) / (
                    high_values[index] - low_values[index])
    return data


def validate_network(network: NeuralNet, validation_data: List[List[Union[float, int]]]):
    correct_answers = 0
    for row in validation_data:

        data: List[float] = row[:-1]
        data_class: int = row[-1]

        for _, value in enumerate(data):
            network.input_layer[_].put(value)

        desired_output = convert(data_class)
        actual_output = network.generate_output()

        if actual_output.index(max(actual_output)) == desired_output.index(1):
            correct_answers += 1

    success_rate = correct_answers / len(validation_data) * 100
    return success_rate, correct_answers


def sgd(training_data: List[List[Union[float, int]]],
        validation_data: List[List[float]],
        network_shape: Tuple[int, int, int],
        epochs,
        lr,
        learning_rate_factor):
    """
    This is the stochastic gradient decent function, it trains the neural network.
    :param training_data: the whole training_dataset
    :param validation_data:
    :param network_shape:
    :param epochs: amount of times the network is trained.
    :param lr: the learning rate.
    :return:
    """

    network: NeuralNet = NeuralNet(train_set=training_data, network_shape=network_shape,
                                   learning_rate=lr)

    for _ in range(epochs):
        for __, row in enumerate(training_data):
            data: List[float] = row[:-1]
            data_class: int = row[-1]
            [network.input_layer[i].put(value) for i, value in enumerate(data)]
            network.forward_propagate(data_class)
            network.back_propagate()
        score, correct_answers = validate_network(network, validation_data)
        print(f"Epoch #{_}, "
              f"Guesses correct:{correct_answers}/{len(validation_data)} - score:{score} - lr: {network.learning_rate}")
        network.learning_rate *= learning_rate_factor  # making the adjustments smaller as epochs go on.


if __name__ == '__main__':
    # fetch dataset
    iris = fetch_ucirepo(id=53)

    # data (as pandas dataframes)
    X: DataFrame = iris.data.features
    y: DataFrame = iris.data.targets

    # properly convert the sets to a list
    x_as_list: List[List[float]] = X.values.tolist()

    # 0 = setosa 1 = versicolor 2 = virginica
    y_as_list = [0 if item == 'Iris-setosa' else 1 if item == 'Iris-versicolor' else 2 for item in
                 list(chain(*y.values.tolist()))]

    # adding classes to the features
    for index, features in enumerate(x_as_list):
        x_as_list[index] = x_as_list[index] + [y_as_list[index]]

    dataset: List[List[float]] = x_as_list

    # randomize the data before converting, easier this way
    random.shuffle(dataset)

    minval, maxval = find_high_low_values(dataset)
    dataset = normalize(dataset, minval, maxval)

    split = int(len(dataset) * .85)
    train_set: List[List[float]] = dataset[:split]
    validate_set: List[List[float]] = dataset[split:]

    # network_shape = n neurons per layer, n layers, n outputs
    sgd(training_data=train_set, validation_data=validate_set, network_shape=(4, 1, 3), epochs=2000, lr=.05,
        learning_rate_factor=0.99995)
