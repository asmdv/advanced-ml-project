import torch
import matplotlib.pyplot as plt
import pickle
def load_data(pt_data_path):
    with open(pt_data_path, 'rb') as file:
        # Load the object from the file
        pt_data = pickle.load(file)
    # pt_data = torch.load(pt_data_path)
    return pt_data

GLOBAL_COLORS = ["#7eb0d5", "#fd7f6f", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7", "#991f17"]

def plot_error_from_data(pt_data, show=False, save_path=None):
    errors = pt_data['errors']
    expansion_epochs = pt_data['expansion_epochs']
    average_errors_across_tasks = []
    errors_per_task = [[] for _ in range(len(errors[0]))]
    for epoch_data in errors:
        average_errors_across_tasks.append(sum(epoch_data) / len(epoch_data))
        for task_index, error in enumerate(epoch_data):
            errors_per_task[task_index].append(error)

    num_epochs = len(errors)

    plt.figure(figsize=(14, 8))

    for task_idx, task_errors in enumerate(errors_per_task):
        plt.plot(range(0, len(task_errors)), task_errors, label=f'Task {task_idx + 1}', linestyle='-', color=GLOBAL_COLORS[task_idx + 1])

    plt.plot(range(0, num_epochs), average_errors_across_tasks, label='Average', marker='o', linewidth=2, color=GLOBAL_COLORS[0])


    for expansion_epoch in expansion_epochs:
        plt.vlines(expansion_epoch, 0, 100, colors=GLOBAL_COLORS[-1], linestyles='--')


    plt.xlabel('Epochs')
    plt.ylabel('Error (%)')
    # plt.title('Average Error across All Tasks and Individual Task Errors over Epochs')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xticks(range(0, num_epochs))

    plt.xticks(fontsize=6)
    if save_path:
        plt.savefig(f"{save_path}/errors.png")
    if show:
        plt.show()

def plot_local_batch_error_from_data(pt_data, show=False, save_path=None):
    errors = pt_data['test_batch_error']
    batches_in_epoch = pt_data['batches_in_epoch']
    print("Batches in epoch: ", batches_in_epoch)
    expansion_epochs = pt_data['expansion_epochs']
    average_errors_across_tasks = []
    errors_per_task = [[] for _ in range(len(errors[0]))]
    for epoch_data in errors:
        average_errors_across_tasks.append(sum(epoch_data) / len(epoch_data))
        for task_index, error in enumerate(epoch_data):
            errors_per_task[task_index].append(error)

    num_epochs = len(errors)

    plt.figure(figsize=(14, 8))

    if batches_in_epoch:
        c = batches_in_epoch
        while (c < len(average_errors_across_tasks)):
            plt.vlines(c, 0, 100, colors='gray', linestyles='--', alpha=0.2)
            c += batches_in_epoch

    for task_idx, task_errors in enumerate(errors_per_task):
        plt.plot(range(1, len(task_errors) + 1), task_errors, label=f'Task {task_idx + 1}', linestyle='-', color=GLOBAL_COLORS[task_idx+1])

    plt.plot(range(1, num_epochs + 1), average_errors_across_tasks, label='Average', linewidth=2, color=GLOBAL_COLORS[0])

    for expansion_epoch in expansion_epochs:
        plt.vlines(expansion_epoch * batches_in_epoch, 0, 100, colors=GLOBAL_COLORS[-1], linestyle='--')

    plt.xlabel('Mini-batch')
    plt.ylabel('Error (%)')
    # plt.title('Average Error across All Tasks and Individual Task Errors over Epochs')
    plt.legend(loc='upper right')
    # plt.grid(True)
    # plt.xticks(range(1, num_epochs + 1))

    plt.xticks(fontsize=6)
    if save_path:
        plt.savefig(f"{save_path}/batch-errors.png")
    if show:
        plt.show()

def main():
    filnames = ["/Users/asif/progs/02-uni/08-advanced-ml-project/DCO/Continual Learning/Baselines/experiments/exp_permuted_mnist_sgd_n_tasks_4_epochs_5_5_threshold_6.0_max_layers_5_freeze_layers_2_0_0_rb_None_2024-04-28_22_23_31/checkpoint.pt"]
    for file in filnames:
        # data_path = f'/Users/asif/Desktop/critical/experiments_ewc_mnists/{file}/checkpoint.pt'
        pt_data = load_data(file)
        plot_error_from_data(pt_data, show=True)
        # plot_local_batch_error_from_data(pt_data, show=True)

if __name__ == '__main__':
    main()
