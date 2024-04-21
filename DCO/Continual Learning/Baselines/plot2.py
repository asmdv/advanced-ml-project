import torch
import matplotlib.pyplot as plt


data_path = '/Users/asif/progs/02-uni/08-advanced-ml-project/DCO/Continual Learning/Baselines/res-checkpoint-permuted_mnist-sgd-0-tasks-5.pt'
data = torch.load(data_path)
sgd_data = data['error']


average_errors_across_tasks = []
errors_per_task = [[] for _ in range(5)]  


for epoch_data in sgd_data:
    if len(epoch_data) == 5:
        
        average_errors_across_tasks.append(sum(epoch_data) / len(epoch_data))
        
        for task_index, error in enumerate(epoch_data):
            errors_per_task[task_index].append(error)
    else:
        
        average_errors_across_tasks.append(epoch_data[0])
        for task_errors in errors_per_task:
            task_errors.append(epoch_data[0])

num_epochs = len(sgd_data)

plt.figure(figsize=(14, 8))

plt.plot(range(1, num_epochs + 1), average_errors_across_tasks, label='Average across all tasks', marker='o', linewidth=2)

for task_idx, task_errors in enumerate(errors_per_task):
    plt.plot(range(1, len(task_errors) + 1), task_errors, label=f'Task {task_idx + 1}', linestyle='-')

plt.xlabel('Epochs')
plt.ylabel('Error (%)')
plt.title('Average Error across All Tasks and Individual Task Errors over Epochs')
plt.legend(loc='upper right')
plt.grid(True)
plt.xticks(range(1, num_epochs + 1))

plt.xticks(fontsize=6)

plt.show()
