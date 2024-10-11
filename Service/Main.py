import pandas as pd
import csv
import random

from .Neuron import neuron

def read_img_to_matrix(file_path, delimiter=';'):
    with open(file_path, 'r') as file:
        # Read lines, split by delimiter, and convert to float
        lines = file.readlines()
        last_two_values = []
        arr = []
        for line in lines:
            # Split line into values and ignore comments
            values = line.split(delimiter)


            arr.append([float(value) for value in values[:-2]])
            last_two_values.append([str(values[-2]), int(values[-1])])

        matrix = pd.DataFrame(arr)
        return matrix, last_two_values


def read_scales_to_matrix(file_path, delimiter=';'):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        arr = []
        for line in lines:
            values = line.split(delimiter)
            arr.append([float(value) for value in values])
        matrix = pd.DataFrame(arr)
        return matrix


def create_scales_file(file_name, num_rows):
    # Создаем и открываем файл для записи
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)

        for _ in range(num_rows):
            value = random.uniform(0, 1)
            writer.writerow([value])


create_scales_file('scales_1.csv', 10)


def main(epochs, res_the_scales = False):
    # Load image data from img.csv
    img_matrix = read_img_to_matrix('./Files/data_1.csv')

    if res_the_scales:
        create_scales_file(f'scales_1.csv', 10)

    for _ in range(epochs):
        if img_matrix is not None:
            # Assume the neuron function requires layers of weights
            layer_matrices = []

            # for i in range(len(img_matrix)):

            # Attempt to load scales files into matrix
            layer_index = 1
            while True:
                try:
                    # Construct the filename based on the layer index
                    file_path = f'scales_{layer_index}.csv'
                    layer_matrix = read_scales_to_matrix(file_path)

                    if layer_matrix is not None:
                        layer_matrices.append(layer_matrix)
                    else:
                        break

                    # Call the neuron function with the image matrix and this layer's matrix
                    neuron(layer_matrix)

                    # Increment to next layer's scales file
                    layer_index += 1

                except FileNotFoundError:
                    # Break loop if file doesn't exist
                    print(f"No more scales files found after {file_path}.")
                    break




if __name__ == "__main__":
    file_path = "../Files/data_1.csv"


    matrix = read_img_to_matrix(file_path)
    print(matrix)