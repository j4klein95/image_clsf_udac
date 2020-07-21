import json
import argparse
from collections import OrderedDict

import numpy as np

import torch

from torchvision import transforms, models

from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/home/workspace/aipnd-project/flowers/test/1/image_06743.jpg',
                        help='Image to classify.')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Path to model checkpoint')
    parser.add_argument('--topk', type=int, default='5', help="Number of top guesses for the classification")
    parser.add_argument('--category_names', type=str, default='/home/workspace/aipnd-project/flower_to_name.json',
                        help='Json file with path to flower labels')

    return parser.parse_args()


def load_model_checkpoint(path):
    state = torch.load(path)
    model = getattr(models, state['arch'])(pretrained=True)

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


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()

    if device == "cuda":
        model.cuda()
    else:
        model.cpu()

    pic_array = process_image(image_path)
    pic_tensor = torch.from_numpy(pic_array)

    if device == "cuda":
        inputs = pic_tensor.float().cuda()
        inputs = inputs.unsqueeze(0)
    else:
        inputs = pic_tensor.float()
        inputs = inputs.unsqueeze(0)

    output = model.forward(inputs)
    top_p = torch.exp(output).data.topk(topk)
    probs = top_p[0].cpu()
    classes = top_p[1].cpu()
    class_nw = {model.class_to_idx[k]: k for k in model.class_to_idx}

    class_tags = []
    for label in classes.numpy()[0]:
        class_tags.append(class_nw[label])

    return probs.numpy()[0], class_tags


def main():
    _args = get_args()

    model_from_chkpt = load_model_checkpoint(_args.checkpoint)
    processed_image = process_image(_args.input)
    probs, classif = predict(processed_image, model_from_chkpt, _args.topk)

    with open(_args.category_names, 'r') as labels:
        flower_names = json.load(labels)

    print(probs, classif)


if __name__ == '__main__':
    main()
