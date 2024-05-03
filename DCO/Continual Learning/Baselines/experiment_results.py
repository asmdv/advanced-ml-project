import os
import plotter
import pandas as pd
def find_checkpoint_files(folder_path):
    checkpoint_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == "final.pt":
                checkpoint_files.append(os.path.join(root, file))
    return checkpoint_files

folder_path = "/Users/asif/Desktop/untitled_f"
checkpoint_files = find_checkpoint_files(folder_path)

table = {'name': [],
        'avg': [],
        'task1': [],
        'task2': [],
        'task3': [],
        'task4': [],
        'task5': []
        }

if checkpoint_files:
    print("Found final files:")
    for final_file in checkpoint_files:
        print(f"{final_file}")
        data = plotter.load_data(final_file)
        errors = plotter.get_errors(data)
        table['name'].append(os.path.dirname(final_file).split('/')[-1])
        table['avg'].append(errors['average_error'][-1])
        for i, value in enumerate(errors['task_error']):
            table[f'task{i+1}'].append(value[-1])
    df = pd.DataFrame(table)
    df.to_csv('result.csv', index=False)

else:
    print("No final files found in the specified folder and its subdirectories.")
