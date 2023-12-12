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
from typing import List, Union, Tuple

from pandas import DataFrame
from ucimlrepo import fetch_ucirepo

from Network import NeuralNet, convert


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
        lr):
    """
    This is the stochastic gradient decent function, it trains the neural network.
    :param training_data: the whole training_dataset
    :param validation_data:
    :param network_shape:
    :param epochs: amount of times the network is trained.
    :param lr: the learning rate.
    :return:
    """

    network: NeuralNet = NeuralNet(train_set=training_data, validation_set=validation_data, network_shape=network_shape,
                                   learning_rate=lr)

    for _ in range(epochs):
        # random.shuffle(training_data)
        for __, row in enumerate(training_data):
            data: List[float] = row[:-1]
            data_class: int = row[-1]
            [network.input_layer[i].put(value) for i, value in enumerate(data)]
            network.forward_propagate(data_class)
            network.back_propagate()
        score, correct_answers = validate_network(network, validation_data)
        network.learning_rate *= 0.995
        print(score)

def MBGD(training_data: List[List[Union[float, int]]],
        validation_data: List[List[float]],
        network_shape: Tuple[int, int, int],
        batch_size: int,
        epochs: int,
        lr: float):
    """
    Mini batch gradient descent, slower than SGD, but more accurate.
    @param training_data:
    @param validation_data:
    @param network_shape:
    @param batch_size:
    @param epochs:
    @param lr:
    @return:
    """
    network: NeuralNet = NeuralNet(train_set=training_data, validation_set=validation_data, network_shape=network_shape,
                                   learning_rate=lr)

    for _ in range(epochs):
        # random.shuffle(training_data)
        for __, row in enumerate(training_data):
            data: List[float] = row[:-1]
            data_class: int = row[-1]
            [network.input_layer[i].put(value) for i, value in enumerate(data)]
            network.forward_propagate(data_class)
            network.back_propagate()
        score, correct_answers = validate_network(network, validation_data)
        network.learning_rate *= 0.995
        print(score)


if __name__ == '__main__':
    # fetch dataset
    iris = fetch_ucirepo(id=53)

    # data (as pandas dataframes)
    X: DataFrame = iris.data.features
    y: DataFrame = iris.data.targets

    shuffled_x: DataFrame = X.sample(frac=1, random_state=1)
    shuffled_y: DataFrame = y.sample(frac=1, random_state=1)

    # properly convert the sets to a list
    x_as_list = shuffled_x.values.tolist()

    # 0 = setosa 1 = versicolor 2 = virginica
    y_as_list = [0 if item == 'Iris-setosa' else 1 if item == 'Iris-versicolor' else 2 for item in
                 list(chain(*shuffled_y.values.tolist()))]

    # adding classes to the features
    for index, features in enumerate(x_as_list):
        x_as_list[index] = x_as_list[index] + [y_as_list[index]]

    dataset = x_as_list
    split = int(len(dataset) * .8)
    train_set: List[List[float]] = dataset[:split]
    validate_set: List[List[float]] = dataset[split:]
    # network_shape = n neurons per layer, n layers, n outputs
    sgd(training_data=train_set, validation_data=validate_set, network_shape=(4, 5, 3), epochs=1000, lr=0.5)
