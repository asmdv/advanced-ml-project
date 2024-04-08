import torch
import matplotlib.pyplot as plt

def try_plot(data):
    if isinstance(data, (list, torch.Tensor)):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        plt.figure(figsize=(10, 6))
        plt.plot(data)
        plt.title("Data Plot")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()
    else:
        print("Data is not in a directly plottable format.")

#Please replace 'C:\Users\Dell\Downloads\res-permuted_mnist-dco-0.pt' with your .pt file path

data = torch.load(r'C:\Users\Dell\Documents\GitHub\advanced-ml-project\DCO\Continual Learning\Baselines\res-checkpoint-permuted_mnist-sgd-0-tasks-5.pt')

#change 'dco' with the algo used
if 'sgd' in data:
    ewc_data = data['sgd']
    #print(f"Processing ... Type: {type(ewc_data)}")
    
    if torch.is_tensor(ewc_data):
        
        try_plot(ewc_data)
    elif isinstance(ewc_data, dict):
        
        for key, value in ewc_data.items():
            print(f"Key: {key}, Value Type: {type(value)}")
            try_plot(value)
    elif isinstance(ewc_data, list):
        
        try_plot(ewc_data)
    else:
        print("The 'ewc' data type is not directly supported for plotting.")
else:
    print("The key 'ewc' was not found in the loaded data.")
