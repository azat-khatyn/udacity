import numpy as np
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import argparse
import json
from PIL import Image


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='File name for checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='Number of returned most probable classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='File name for category/name mappings')
    parser.add_argument('--gpu', type=bool, default=True, help='Use GPU or CPU as the device')
    parser.add_argument('--correct_class_id', type=int, default=-1, help='Correct class id of the input image')
    return parser.parse_args()


def load_model_from_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.classifier.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def read_cat_to_name(filepath):
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name


def process_image(image):
    w, l = image.size
    if w == l:
        w, l = 256, 256
        a, b, c, d = 16, 16, 240, 240
    elif w < l:
        ratio = (1.0 * l) / w
        w, l = 256, int(256 * ratio)
        a, b, c, d = 16, int((l - 224) / 2), 240, int(l - (l - 224) / 2)
    else:
        ratio = (1.0 * w) / l
        w, l = int(256 * ratio), 256
        a, b, c, d = int((w - 224) / 2), 16, int(w - (w - 224) / 2), 240

    image_resized = image.resize((w, l))
    image_cropped = image_resized.crop((a, b, c, d))

    np_image = np.array(image_cropped) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_image = (np_image - mean) / std
    transposed_image = normalized_image.transpose((2, 0, 1))
    return transposed_image


def predict(image_path, model, topk, cat_to_name):
    model.eval()
    model.to(device)
    image = Image.open(image_path)
    preprocessed_image = process_image(image)
    batch_of_one = torch.tensor(np.array([preprocessed_image])).type(torch.FloatTensor)
    batch_of_one.to(device)
    with torch.no_grad():
        model.type(torch.FloatTensor)
        output = model.forward(batch_of_one)
        output = torch.exp(output)
        probs, top_classes = output.topk(topk)
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        prob_arr = probs.data.numpy()[0]
        pred_indexes = top_classes.data.numpy()[0].tolist()
        pred_labels = [idx_to_class[x] for x in pred_indexes]
        pred_class = [cat_to_name[str(x)] for x in pred_labels]
        return pred_class, prob_arr


if __name__ == '__main__':
    print('Parsing arguments...')
    args = get_arguments()
    print('Arguments are: %s' % str(vars(args)))
    print('Loading model from checkpoint...')
    model = load_model_from_checkpoint(args.checkpoint)
    device = 'cuda' if args.gpu else 'cpu'
    print('Using [%s] for computations...' % device)
    print('Reading category/indices file...')
    cat_to_name = read_cat_to_name(args.category_names)
    print('Inferring the classes and probabilities...')
    classes, probs = predict(args.input, model, args.top_k, cat_to_name)
    print("Inferred classes: %s %s" % (classes, probs))
    if args.correct_class_id >= 0:
        print("Correct class is: " + cat_to_name[str(args.correct_class_id)])
