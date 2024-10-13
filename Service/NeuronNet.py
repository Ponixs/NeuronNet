from .Softmax import softmax
from .NetLay import net_lay
from .CRUD_files import read_scales_to_matrix


def neuron_net(layer_matrices, scales_index):
    for j in range(1, scales_index + 1):

        file_path = f'scales_{j}.csv'
        scales_matrix = read_scales_to_matrix(file_path)

        if scales_matrix is not None:
            layer_matrices = net_lay(layer_matrices, scales_matrix)
        else:
            print("Error: No file with neuron weights")

    scales_matrix = read_scales_to_matrix('scales_end.csv')
    layer_matrices = net_lay(layer_matrices, scales_matrix, True)

    result = softmax(layer_matrices)
    return result
