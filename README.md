# ToyExample ImageClassification
This is a toy task on image classification with PyTorch for a seminar on AI.


## Task description
Use the popular [MNIST](https://en.wikipedia.org/wiki/MNIST_database) data set to demonstrate how to implement a neural network for image classification.

Note:
- Use [PyTorch](https://pytorch.org/) for this.
- The python-package [torchvision](https://pypi.org/project/torchvision/) provides the class [torchvision.datasets.MNIST](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html) as a convenient wrapper to the dataset. It can automatically download the dataset from [Yan LeCun's homepage](https://yann.lecun.com/exdb/mnist/) or a mirror, but you can also download the data manually (and still use the class to process the compressed files).
- Use a small model for quick development. I recommend using [MobileNetV3](https://pytorch.org/vision/main/models/mobilenetv3.html). (The prepared [task.py](task.py) already holds the correct data transformations for it.)

## Overview

The purpose of this project is to facilitate learning how to quickly program an image classification with PyTorch. The [task.py](task.py)-file should guide the interested reader through the process by providing a rough structure with hulls of functions (specifying input / output parameters incl. [type hints](https://docs.python.org/3/library/typing.html)), some variable names, and comments TODO-instructions.



## Project Structure
````
ToyExample_ImageClassification
|-- task.py  # This is the sceleton of a possible solution. It breaks the task into subtasks
|-- solution.py  # This is the corresponding full solution of the task **DO NOT START FROM HERE**
|-- LICENSE
|-- README.md
|-- requirements.txt
````

## Quickstart
I recommend using Python 3.11 (or later).
Set up a virtual environment installing all [requirements.txt](requirements.txt).
````shell
python venv .venv  # create virtual environment
source ./.venv/bin/activate  # activate virtual environment (linux)
pip install -r requirements.txt  # install requirements
````
Now you have a virtual environment (i.e. all Python-packages are stored to the hidden folder .venv in this directory.)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Author
 - max-scw
