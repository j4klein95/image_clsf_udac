import json
import argparse
from collections import OrderedDict

import numpy as np

import torch
from torch import nn, optim

from torchvision import datasets, transforms, models

from helpers import train_model, validation


def cl_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='flowers', help='directory with data to run model against')
    parser.add_argument('--save_dir', type=str, default="checkpoint.pth", help='path to save model checkpoints')
    parser.add_argument('--arch', type=str, default='densenet121', choices=['densenet121', 'vgg13', 'vgg16'],
                        help='pretrained models to train with')
    parser.add_argument('--learning_rate', type=float, default='0.003', help='Learning rate for model')
    parser.add_argument('--hidden_units', type=int, default='256', help='units in layer')
    parser.add_argument('--epochs', type=int, default='5', help='epochs')

    return parser.parse_args()


def main():
    _args = cl_args()

    train_dir = _args.data_dir + '/train'
    valid_dir = _args.data_dir + '/valid'
    test_dir = _args.data_dir + '/test'

    num_batch = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"

    norm_means = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=norm_means, std=norm_std)
                                           ])

    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=norm_means, std=norm_std)
                                          ])

    valid_transorms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=norm_means, std=norm_std)
                                          ])

    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transorms)

    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=num_batch, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=num_batch)
    valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=num_batch)

    class_to_idx = train_datasets.class_to_idx

    ####################
    model_selector = {"densenet121": models.densenet121(pretrained=True),
                      "vgg13": models.vgg13(pretrained=True),
                      "vgg16": models.vgg16(pretrained=True)
                      }

    model = model_selector[_args.arch]

    for param in model.parameters():
        param.requires_grad = False

    input_sizes = model.classifier.in_features

    classifier = nn.Sequential(OrderedDict([('input_layer', nn.Linear(input_sizes, _args.hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('Dropout', nn.Dropout()),
                                            ('h1', nn.Linear(_args.hidden_units, 102)),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))

    model.classifier = classifier

    optimizer = optim.Adam(model.classifier.parameters(), lr=_args.learning_rate)
    train_model(model, _args.epochs, optimizer, train_dataloaders, valid_dataloaders)

    ################

    criterion = nn.NLLLoss()
    validation(model, test_dataloaders, device, criterion)

    model.class_to_idx = train_datasets.class_to_idx

    chk_pt_path = 'dns121_flrcls_model.pth'

    strt_chkpt = {'arch': 'densenet121', 'epochs': _args.epochs, 'learning_rate': _args.learning_rate,
                  'classifier': classifier, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                  'class_to_idx': class_to_idx}

    torch.save(strt_chkpt, _args.save_dir)


if __name__ == '__main__':
    main()
