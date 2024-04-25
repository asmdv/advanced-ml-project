import torch
import matplotlib.pyplot as plt

def load_data(pt_data_path):
    pt_data = torch.load(pt_data_path)
    return pt_data

def plot_error_from_data(pt_data, show=False, save_path=None, expansion_epochs=[]):
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
    for expansion_epoch in expansion_epochs:
        plt.vlines(expansion_epoch, plt.gca().get_ylim()[0], plt.gca().get_ylim()[1], label=f'Expansion {expansion_epoch}', colors='red')

    plt.xlabel('Epochs')
    plt.ylabel('Error (%)')
    # plt.title('Average Error across All Tasks and Individual Task Errors over Epochs')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xticks(range(1, num_epochs + 1))

    plt.xticks(fontsize=6)
    if save_path:
        plt.savefig(f"{save_path}/errors.png")
    if show:
        plt.show()

def main():
    filnames = ["exp_permuted_mnist_ewc_n_tasks_5_epochs_10_10_threshold_8.0_max_layers_0_freeze_layers_0_0_0_rb_5_2024-04-23_00_08_34", "exp_permuted_mnist_ewc_n_tasks_5_epochs_10_10_threshold_8.0_max_layers_0_freeze_layers_0_0_0_rb_20_2024-04-23_00_08_37", "exp_permuted_mnist_ewc_n_tasks_5_epochs_10_10_threshold_8.0_max_layers_0_freeze_layers_0_0_0_rb_128_2024-04-23_00_08_39", "exp_permuted_mnist_ewc_n_tasks_5_epochs_10_10_threshold_8.0_max_layers_0_freeze_layers_0_0_0_rb_None_2024-04-23_00_08_32"]
    for file in filnames:
        data_path = f'/Users/asif/Desktop/critical/experiments_ewc_mnists/{file}/checkpoint.pt'
        pt_data = load_data(data_path)
        plot_error_from_data(pt_data, show=True, save_path=f"/Users/asif/Desktop/critical/experiments_ewc_mnists/{file}/plots")


if __name__ == '__main__':
    main()
