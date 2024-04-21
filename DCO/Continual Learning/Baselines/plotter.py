import torch
import matplotlib.pyplot as plt

def load_data(pt_data_path):
    pt_data = torch.load(pt_data_path)
    return pt_data

def plot_error_from_data(pt_data, show=False, save_path=None):
    errors = pt_data['errors']
    average_errors_across_tasks = []
    errors_per_task = [[] for _ in range(len(errors[0]))]
    for epoch_data in errors:
        average_errors_across_tasks.append(sum(epoch_data) / len(epoch_data))
        for task_index, error in enumerate(epoch_data):
            errors_per_task[task_index].append(error)

    num_epochs = len(errors)

    plt.figure(figsize=(14, 8))

    plt.plot(range(1, num_epochs + 1), average_errors_across_tasks, label='Average across all tasks', marker='o', linewidth=2)

    for task_idx, task_errors in enumerate(errors_per_task):
        plt.plot(range(1, len(task_errors) + 1), task_errors, label=f'Task {task_idx + 1}', linestyle='-')

    plt.xlabel('Epochs')
    plt.ylabel('Error (%)')
    # plt.title('Average Error across All Tasks and Individual Task Errors over Epochs')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xticks(range(1, num_epochs + 1))

    plt.xticks(fontsize=6)
    if save_path:
        plt.savefig(f"{save_path}/errors.png")
    plt.show()

def main():
    data_path = '/Users/asif/progs/02-uni/08-advanced-ml-project/DCO/Continual Learning/Baselines/experiments/experiment_permuted_mnist_sgd_n_tasks_5_epochs_1_1_rank_0_2024-04-21_19_28_18/final.pt'
    pt_data = load_data(data_path)
    plot_error_from_data(pt_data, show=True)


if __name__ == '__main__':
    main()
