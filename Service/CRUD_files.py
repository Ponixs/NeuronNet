import pandas as pd
import csv
import random


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


def read_scales_to_matrix(file_name, delimiter=';'):
    with open(f'./Files/scales/{file_name}', 'r') as file:
        lines = file.readlines()
        arr = []
        for line in lines:
            values = line.split(delimiter)
            arr.append([float(value) for value in values])
        matrix = pd.DataFrame(arr)
        return matrix


def create_scales_file(file_name, num_rows, num_columns):
    # Создаем и открываем файл для записи
    with open(f'./Files/scales/{file_name}', mode='w', newline='') as file:
        writer = csv.writer(file)
        for _ in range(num_rows):
            row = [f"{random.uniform(-1, 1):.4f}" for _ in range(num_columns)]
            writer.writerow([';'.join(row)])
