#
# Created by Jippe Heijnen on 04-12-2023.
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
import numpy as np


def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    mse = (1 / (2 * n)) * np.sum(np.square(y_true - y_pred))
    return mse


if __name__ == '__main__':
    # Example data
    y_true = np.array([2, 4, 5, 4, 5])
    y_pred = np.array([1.5, 3.5, 4.5, 4.0, 5.5])

    # Calculate MSE
    mse = mean_squared_error(y_true, y_pred)

    print("Mean Squared Error:", mse)
