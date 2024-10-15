import os
from dotenv import load_dotenv

from Auxiliary import back_propagation
from .NeuronNet import neuron_net
from .CRUD_files import *


def train(epochs, res_the_scales=False):
    img_matrix = read_img_to_matrix('./Files/training.csv')

    load_dotenv()
    const_width = int(os.getenv('const_width'))
    count_neurons1 = int(os.getenv('count_neurons1'))
    count_neurons2 = int(os.getenv('count_neurons2'))
    count_class = int(os.getenv('count_class'))
    scales_index = int(os.getenv('scales_index'))

    if res_the_scales:
        create_scales_file('scales_1.csv', const_width ** 2, count_neurons1)
        create_scales_file('scales_2.csv', count_neurons1, count_neurons2)
        create_scales_file('scales_end.csv', count_neurons2, count_class)

    for _ in range(epochs):
        if img_matrix is not None:

            row = img_matrix[0].shape[0]

            for i in range(row):

                layer_matrices = img_matrix[0].iloc[i]
                layer_matrices = neuron_net(layer_matrices, scales_index)[1]

                true_answer = img_matrix[1][i][0] - 1
                back_propagation(layer_matrices, true_answer)
        print(f"Later {epochs} epochs.")



if __name__ == "__main__":
    file_path = "../Files/data_1.csv"

    matrix = read_img_to_matrix(file_path)
    print(matrix)
