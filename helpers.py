import json

from collections import OrderedDict
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.autograd as Variable
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from PIL import Image


def train_model(model, epochs, optimizer, training_loader, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    steps = 0
    running_loss = 0
    print_every = 20

    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        for inputs, labels in training_loader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(test_loader):.3f}.. "
                      f"Test accuracy: {accuracy / len(test_loader):.3f}")
                running_loss = 0
                model.train()


def validation(model, test_dataloader, device, criterion):
    model.eval()
    model.to(device)
    accuracy = 0
    test_loss = 0

    for inputs, targets in test_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            output = model.forward(inputs)
        test_loss += criterion(output, targets).item()
        ps = torch.exp(output).data

        equality = (targets.data == ps.max(1)[1])
        if device == "cuda":
            accuracy += equality.type_as(torch.cuda.FloatTensor()).mean()
        else:
            accuracy += equality.type_as(torch.FloatTensor()).mean()
    print("Test Loss: {:.3f}".format(test_loss / len(test_dataloader)))
    print("Test Accuracy: {:.3f}".format(accuracy / len(test_dataloader)))


def load_chkpt(path):
    state = torch.load(path)
    model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = state['classifier']
    optimizer.load_state_dict(state['optimizer'])
    model.load_state_dict(state['state_dict'])
    model.class_to_idx = state['class_to_idx']

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image).convert("RGB")

    in_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    image = in_transforms(image)[:3, :, :]
    return image.numpy()


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def view_classify(img, probs, classifications):
    class_labels = [flower_to_name[x] for x in classifications]

    fig = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot2grid((15, 9), (0, 0), colspan=9, rowspan=9)
    ax2 = plt.subplot2grid((15, 9), (9, 2), colspan=5, rowspan=5)

    image = Image.open(img)
    ax1.axis('off')
    ax1.set_title(flower_to_name[label])
    ax1.imshow(image)
    class_labels = []
    for class_ in classifications:
        class_labels.append(flower_to_name[class_])
    y_pos = np.arange(5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_labels)
    ax2.invert_yaxis()  # probabilities read top-to-bottom
    ax2.set_xlabel('Probability')
    ax2.barh(y_pos, probs, xerr=0, align='center')

    plt.show()