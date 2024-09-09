import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
# dataset
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
# optimizer
from torch.optim import Adam
# image processing
from PIL import Image

import copy
from timeit import default_timer



def train_model(
        model: nn.Module,
        dataloader_training: DataLoader,
        n_epochs: int,
        dataloader_validation: DataLoader = None,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        optimizer=None,
        device: str = "cpu"
) -> nn.Module:
    """
    Trains a PyTorch model
    :param model: PyTorch model to train
    :param dataloader_training: dataloader holding the training set
    :param n_epochs: Number of training epochs
    :param dataloader_validation: dataloader holding the validation set (to ensure that the best model is returned later)
    :param criterion: evaluation criterion (for classification usually cross-entropy loss
    :param optimizer: optimizer to calculate the training step, usually Adam
    :param device: usually CPU or GPU
    :return: best trained model
    """
    t0 = default_timer()  # time the execution of the function

    # initialize best_loss and model_weights to keep track of the best model
    best_loss = torch.inf
    best_model_weights = copy.deepcopy(model.state_dict())

    # for every epoch
    for i_epoch in range(n_epochs):
        # TODO: calculate the training loss -> to adjust the weights
        model.train()  # Set model to training mode

        # TODO: calculate the validation loss -> to determine if the training improves or overfitting occurs (keep the best model)
        model.eval()  # Set model to evaluate mode

    time_elapsed = default_timer() - t0
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s. Best val Loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


def build_model(
        n_out: int = 1000,
        freeze_backbone: bool = True
) -> nn.Module:
    """
    Constructs a (small) MobileNetV3 and loads pretrained weights. Freezes the backbone / descriptor if desired.
    :param n_out: Number of output features that the classifier head of the model should learn
    :param freeze_backbone: flag indicating if the parameters of the descriptor backbone should be kept to their pretrained weights
    :return:
    """

    # TODO: construct / load model with pretrained weights
    model =

    if freeze_backbone:
        # TODO: freeze model parameter/weights of descriptor
        pass

    # TODO: adjust number of output features of classifier (if necessary)

    return model


def grayscale_to_color(image: Image) -> Image:
    return image.convert("RGB")


if __name__ == "__main__":
    n_workers = 2
    batch_sz = 32
    n_epochs = 2

    # build model (Modify the classifier to fit MNIST i.e. to 10 classes)
    mobilenet = build_model(10)

    # corresponding transforms for MobileNet
    transform = transforms.Compose([
        transforms.Lambda(grayscale_to_color),  # Add channel expansion
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the MNIST dataset training & test split
    dataset_training = MNIST(root='./data', train=True, transform=transform, download=True)
    dataset_test = MNIST(root='./data', train=False, transform=transform, download=True)

    # TODO: training / validation / test split
    # NOTE: you may want to reduce the number of points used for the split to 1% of the dataset for development purposes

    # TODO: create dataloaders for the training, validation and test split
    dataloader_train=
    dataloader_val=
    dataloader_test=

    # Model Training
    # Evaluation criterion. For classification usually cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # Set device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Train model
    mdl = train_model(
        mobilenet,
        dataloader_training=dataloader_train,
        dataloader_validation=dataloader_val,
        n_epochs=n_epochs,
        criterion=criterion,
        device=device
    )

    # TODO: calculate the loss of the trained model on the test set

    # TODO: print result
    print(f"Test loss: {}")
