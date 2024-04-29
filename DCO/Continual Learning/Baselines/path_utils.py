import string
import random
import os
import shutil
import datetime
class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()

def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it doesn't exist.

    Parameters:
    directory_path (str): Path of the directory to be created.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except OSError as err:
            print(f"Error: Creating directory '{directory_path}' - {err}")
    else:
        print(f"Directory '{directory_path}' already exists.")




def generate_random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))

def overwrite_directory(directory_path):
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
        except OSError as err:
            print(f"Error: Unable to remove directory '{directory_path}' - {err}")
    try:
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    except OSError as err:
        print(f"Error: Creating directory '{directory_path}' - {err}")

def create_experiment_path(args):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H_%M_%S")
    freeze_name = "freeze" if args.freeze else "no-freeze"
    experiment_name = f"experiments/exp_{args.cl_dataset}_{args.cl_method}_n_tasks_{args.num_tasks}_epochs_{args.lr_epochs}_{args.cl_epochs}_threshold_{args.cl_error_threshold}_max_layers_{args.max_allowed_added_layers}_{freeze_name}_layers_{'_'.join(str(x) for x in args.added_layer_conf)}_rb_{args.replay_buffer_batch_size}_{formatted_time}"
    create_directory_if_not_exists(experiment_name)
    create_directory_if_not_exists(f"{experiment_name}/plots")
    return experiment_name
