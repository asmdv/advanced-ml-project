import matplotlib.pyplot as plt
import re

def parse_log_file(filepath):
    data = {}
    with open(filepath, 'r') as file:
        for line in file:
            match = re.search(r'Epoch (\d+) \| Task\s+(\d+) => Average TEST Loss: ([\d.]+),', line)
            if match:
                epoch = int(match.group(1))
                task = f"Task {match.group(2)}"
                error = float(match.group(3))
                
                if task not in data:
                    data[task] = {'Epochs': [], 'Error': []}
                data[task]['Epochs'].append(epoch)
                data[task]['Error'].append(error)
    return data

def plot_data(data):
    plt.figure(figsize=(15, 10))
    for task, metrics in data.items():
        plt.plot(metrics['Epochs'], metrics['Error'], label=task)

    plt.title('Error Rate Across Epochs for Each Task')
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

log_file_path = r'C:\Users\Dell\Downloads\a1.txt'
parsed_data = parse_log_file(log_file_path)
plot_data(parsed_data)
