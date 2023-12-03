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
from Network import Neuron

if __name__ == '__main__':

    # setting inputs
    x0 = 1
    x1 = 0

    # adder network
    n0 = Neuron(inputs=[(x0, -0.5), (x1, -0.5)], thresh_value=-0.5)
    n1 = Neuron(inputs=[(x0, -0.5), (n0, -0.5)], thresh_value=-0.5)
    n2 = Neuron(inputs=[(n0, -0.5), (x1, -0.5)], thresh_value=-0.5)
    n3_carry = Neuron(inputs=[(n0, -0.5)], thresh_value=0)
    n4_sum = Neuron(inputs=[(n1, -0.5), (n2, -0.5)], thresh_value=-0.5)

    print(f"With x0 = {x0} and x1 = {x1}\n"
          f"carry = {n3_carry.calc_output()} and sum = {n4_sum.calc_output()}")
