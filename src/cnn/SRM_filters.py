from typing import Dict

import numpy as np
from torch import Tensor, stack


def get_filters():
    """
    Function that return the required high pass SRM filters for the first convolutional layer of our implementation
    :return: A pytorch Tensor containing the 30x3x5x5 filter tensor with type
    [number_of_filters, input_channels, height, width]
    """

    filters: Dict[str, Tensor] = {}

    # 1st Order
    filters["1O1"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0]]))
    filters["1O2"] = Tensor(np.rot90(filters["1O1"]).copy())
    filters["1O3"] = Tensor(np.rot90(filters["1O2"]).copy())
    filters["1O4"] = Tensor(np.rot90(filters["1O3"]).copy())
    filters["1O5"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0]]))
    filters["1O6"] = Tensor(np.rot90(filters["1O5"]).copy())
    filters["1O7"] = Tensor(np.rot90(filters["1O6"]).copy())
    filters["1O8"] = Tensor(np.rot90(filters["1O7"]).copy())
    # 2nd Order
    filters["2O1"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0]]))
    filters["2O2"] = Tensor(np.rot90(filters["2O1"]).copy())
    filters["2O3"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -2, 0, 0], [0, 0, 0, 1, 0],
                                      [0, 0, 0, 0, 0]]))
    filters["2O4"] = Tensor(np.rot90(filters["2O3"]).copy())
    # 3rd Order
    filters["3O1"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, -3, 1, 0], [0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0]]))
    filters["3O2"] = Tensor(np.rot90(filters["3O1"]).copy())
    filters["3O3"] = Tensor(np.rot90(filters["3O2"]).copy())
    filters["3O4"] = Tensor(np.rot90(filters["3O3"]).copy())
    filters["3O5"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, -3, 0, 0], [0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0]]))
    filters["3O6"] = Tensor(np.rot90(filters["3O5"]).copy())
    filters["3O7"] = Tensor(np.rot90(filters["3O6"]).copy())
    filters["3O8"] = Tensor(np.rot90(filters["3O7"]).copy())
    # 3x3 SQUARE
    filters["3x3S"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0],
                                       [0, 0, 0, 0, 0]]))
    # 3x3 EDGE
    filters["3x3E1"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]]))
    filters["3x3E2"] = Tensor(np.rot90(filters["3x3E1"]).copy())
    filters["3x3E3"] = Tensor(np.rot90(filters["3x3E2"]).copy())
    filters["3x3E4"] = Tensor(np.rot90(filters["3x3E3"]).copy())
    # 5X5 EDGE
    filters["5x5E1"] = Tensor(np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                                        [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))
    filters["5x5E2"] = Tensor(np.rot90(filters["5x5E1"]).copy())
    filters["5x5E3"] = Tensor(np.rot90(filters["5x5E2"]).copy())
    filters["5x5E4"] = Tensor(np.rot90(filters["5x5E3"]).copy())
    # 5x5 SQUARE
    filters["5x5S"] = Tensor(np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                                       [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]))

    return vectorize_filters(filters)


def vectorize_filters(filters: dict):
    tensor_list = []

    w = list(filters.values())

    for i in range(1, 31):
        tmp = []
        k = ((i - 1) % 10) + 1
        tmp.append(w[3 * k - 3])
        tmp.append(w[3 * k - 2])
        tmp.append(w[3 * k - 1])
        tensor_list.append(stack(tmp))

    tensor = stack(tensor_list)

    # ---- SRM NORMALIZATION ----
    tensor = tensor - tensor.mean(dim=(2, 3), keepdim=True)
    tensor = tensor / (tensor.abs().sum(dim=(2, 3), keepdim=True) + 1e-8)

    return tensor
