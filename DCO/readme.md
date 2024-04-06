# Codes

## Description
We provide the source codes used to generate the experimental results from our paper *Overcoming Catastrophic Forgetting via Direction-Constrained Optimization*.


## Reproduction

1. To reproduce the experimental results for *Sectoin 3 (Loss Lanscape)*, please run `python cone_mnist.py` and `python cone_cifar10.py`. 
   
   The plots will show up in *illustrative_example* folder and *illustrative_example_cifar10* folder for MNIST dataset and CIFAR-10 dataset respectively.

2. To reproduce the experimental results for *Sectoin 5 (Continual Learning)*, please see *Baselines* folder and *DCO-COMP* folder. 
   * *Baselines* folder: use `run_fashion_mnist.sh`, `run_mnist.sh` and `run_cifar100.sh` to generate the baseline results (including DCO) for fashion mnist dataset, mnist dataset and cifar dataset.
   * *DCO-COMP* folder: use `run_all_datasets.sh` to generate the DCO-COMP results on all datasets.
   
Once the bash file starts to be run, the intermediate results could be visualized in Visdom (https://github.com/facebookresearch/visdom).


## Implementation

Most details can be found in `main_[dataset].py`. Detailed comments are provided in the codes to help the readers understand the structure and logic of the codes. The usage of each file is listed as follows:

* `main_[dataset].py`: the major file which contains all continual learning methods

* `data.py`: define how we generate and process each continual learning dataset

* `models.py`: define all architectures we use for the experiments

* `trainer.py`: define training and testing functions

* `algorithm.py`: define the algorithm for training the linear autoenocder

* `utils.py`: provide handy tools for other functions

* `options.py`: provide options for hyperparameter search

*The codes were inspired by open source codes [A-GEM](https://github.com/facebookresearch/agem). We made necessary revision and tested the codes on a four-gpu machine.*