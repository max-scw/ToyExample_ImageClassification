import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

# dataset
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

# optimizer
from torch.optim import Adam

from PIL import Image

from tqdm import tqdm
from timeit import default_timer
import copy

from typing import Union


# Training function
def train_model(
    model: nn.Module,
    dataloader_training: DataLoader,
    n_epochs: int,
    dataloader_validation: DataLoader = None,
    criterion=nn.Module,
    optimizer=None,
    device: Union[str, int] = "cpu",
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
    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=0.001)

    t0 = default_timer()  # time the execution of the function

    # initialize best_loss and model_weights to keep track of the best model
    best_loss = torch.inf
    best_model_weights = copy.deepcopy(model.state_dict())
    history = []

    for i_epoch in range(n_epochs):
        desc = {}
        # Each epoch has a training and validation phase
        for phase in ["training", "validation"]:
            is_training = phase == "training"

            if is_training:
                model.train()  # Set model to training mode
                dataloader = dataloader_training
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = dataloader_validation if dataloader_validation else dataloader_training

            running_loss = 0.0
            running_len = 0

            # Iterate over data.
            # progress bar object
            pbar = tqdm(dataloader, desc=f"Epoch {i_epoch + 1}/{n_epochs}, {phase}")
            for inputs, labels in pbar:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history only in train
                with torch.set_grad_enabled(is_training):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if is_training:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_len += len(labels)

                # update postfix of the progress bar
                pbar.set_postfix({"loss": running_loss / running_len})

            # calculate loss
            epoch_loss = running_loss / len(dataloader)
            # store information to update the bar
            desc[f"{phase}_loss"] = epoch_loss

            # deep copy the model
            if not is_training and (epoch_loss < best_loss):
                best_loss = epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())

        history.append(desc)

    time_elapsed = default_timer() - t0
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s. Best val Loss: {best_loss:4f}")

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model, history


def grayscale_to_color(image: Image) -> Image:
    return image.convert("RGB")


def build_model(n_out: int = 1000, freeze_backbone: bool = True) -> nn.Module:
    """Constructs a (small) MobileNetV3 and loads pretrained weights. Freezes the backbone / descriptor if desired."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # freeze model parameter/weights of descriptor
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # modify number of output features if necessary
    if n_out != model.classifier[-1].out_features:
        model.classifier[-1].out_features = n_out

    # Ensure that the classifier's parameters will be updated
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


if __name__ == "__main__":
    n_workers = 2
    batch_sz = 32
    n_epochs = 2

    # build model (Modify the classifier to fit MNIST i.e. to 10 classes)
    mobilenet = build_model(10)

    # corresponding transforms for MobileNet (extended by a channel expansion because MNIST provides grayscale images
    # but MobileNet expects inputs with 3 channels)
    transform = transforms.Compose([
        transforms.Lambda(grayscale_to_color),  # Add channel expansion
        models.MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    ])

    # Load the MNIST dataset training & test split
    dataset_training = MNIST(root="./data", train=True, transform=transform, download=True)
    dataset_test = MNIST(root="./data", train=False, transform=transform, download=True)

    # train / test / validation split
    # NOTE: you may want to reduce the number of points used for the split to 1% of the dataset for development purposes
    n_points_all = len(dataset_training)
    # reduce number of points
    n_points = int(0.01 * n_points_all)  # FIXME: held_out 99%% of the data for faster training
    # Define the split ratio (e.g., 80% for training, 20% for validation)
    train_size = int(0.8 * n_points)
    val_size = n_points - train_size

    # Split the dataset
    train_subset, val_subset, _ = random_split(dataset_training, [train_size, val_size, n_points_all - n_points])

    # Create DataLoaders for the training and validation sets
    dataloaders = dict()
    for ky in ["training", "validation", "test"]:
        shuffle = False

        if ky == "validation":
            data = val_subset
        elif ky == "training":
            data = train_subset
            shuffle = True
        else:
            data = dataset_test
        # create dataloader
        dataloaders[ky] = DataLoader(dataset=data, batch_size=batch_sz, shuffle=shuffle, num_workers=n_workers)

    # train model
    # Set device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    criterion = nn.CrossEntropyLoss()

    mdl, hist = train_model(
        mobilenet,
        dataloader_training=dataloaders["training"],
        dataloader_validation=dataloaders["validation"],
        n_epochs=n_epochs,
        criterion=criterion,
        device=device,
    )

    running_loss = 0.0

    mdl.eval()
    for inputs, labels in tqdm(dataloaders["test"], desc="Testing"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = mdl(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)

    print(f"Test loss: {running_loss / len(dataloaders['test'])}")
