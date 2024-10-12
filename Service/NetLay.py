import pandas as pd

from .Sigmoid import sigmoid
from .CrossEntropy import binary_cross_entropy


def net_lay(img_matrix: pd.DataFrame, scales_matrix: pd.DataFrame):
    col = scales_matrix.shape[1]

    arr = []
    for j in range(col):
        var = img_matrix * scales_matrix.iloc[:, j]
        arr.append(sigmoid(var.sum()))
    y = pd.DataFrame(arr)

    return y
