import torch
import matplotlib.pyplot as plt
import numpy as np


def try_plot(data, title="Data Plot"):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

# Put your file path here
file_path = "res-final-permuted_mnist-sgd-0-tasks-5.pt"
pt_load = torch.load(fr'{file_path}')


def column(matrix, i):
    new = []
    for row in matrix:
        print(row)
        new.append(row[i])
    return new


# change 'dco' with the algo used
if 'sgd' in pt_load:
    global_data = pt_load['sgd']
    # print(f"Processing ... Type: {type(ewc_data)}")

    if torch.is_tensor(global_data):

        try_plot(global_data)
    elif isinstance(global_data, dict):
        for key, value in global_data.items():
            print(f"Key: {key}, Value Type: {type(value)}")
            try_plot(value)
    elif isinstance(global_data, list):
        _data = np.array([row for row in global_data if len(row) == 5])
        for i in range(_data.shape[1]):
            try_plot(_data[:, i])
        # print(_data)
        # print(np.mean(_data, axis=1))
        try_plot(np.mean(_data, axis=1), "Average Error")
    else:
        print("The 'ewc' data type is not directly supported for plotting.")
else:
    print("The key 'ewc' was not found in the loaded data.")