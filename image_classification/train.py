import numpy as np
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import argparse
from collections import OrderedDict


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory with the input dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='File name for checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', help='Type of preloaded model, e.g. vgg16')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=1024, help='Number of units in the hidden layer')
    parser.add_argument('--epochs', type=int, default=100, help='Number of iterations during learning')
    parser.add_argument('--gpu', type=bool, default=True, help='Use GPU or CPU as the device')
    return parser.parse_args()


def get_directories(data_dir):
    return data_dir + '/train', data_dir + '/valid', data_dir + '/test'


def create_dataloaders(train_dir, valid_dir, test_dir):
    means = [0.485, 0.456, 0.406]
    stgs = [0.229, 0.224, 0.225]
    data_transforms = [transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, stgs)]),
                       transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, stgs)]),
                       transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, stgs)])
                       ]
    image_datasets = [datasets.ImageFolder(train_dir, transform=data_transforms[0]),
                      datasets.ImageFolder(valid_dir, transform=data_transforms[1]),
                      datasets.ImageFolder(test_dir, transform=data_transforms[2])]
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[1], batch_size=64),
                   torch.utils.data.DataLoader(image_datasets[2], batch_size=64)]
    return dataloaders


def build_model(arch, hidden_units):
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        raise Exception('Unknown model! Choose one of the models: vgg11, vgg13, vgg16, vgg19!')
    print('Base model is [%s]' % arch)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu1', nn.ReLU(inplace=True)),
        ('drop1', nn.Dropout(0.2)),
        ('fc3', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    print('Classifier of the model is %s' % model.classifier)
    return model


def train_model(model, train_dataloader, val_dataloader, learning_rate, epochs, device):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.class_to_idx = train_dataloader.dataset.class_to_idx
    model.to(device)

    running_loss = 0.0
    print_every = 5
    final_train_loss = 0.0
    current_test_acc = 0.0
    for i, (inputs, labels) in enumerate(train_dataloader):
        epoch = i + 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        _, predictions = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if epoch % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            final_train_loss = running_loss / print_every
            current_test_acc = accuracy / len(val_dataloader)
            print(f"Epoch {epoch}/{epochs} "
                  f"Train loss: {final_train_loss:.3f} "
                  f"Test loss: {test_loss / len(val_dataloader):.3f} "
                  f"Test accuracy: {current_test_acc:.3f}")
            running_loss = 0
            model.train()

        if epoch >= epochs:
            break
        if current_test_acc >= 0.75:
            print('Reached accuracy of 0.75; breaking the training off!')
            break

    return final_train_loss, criterion


def validate_model(model, criterion, test_dataloader, device):
    current_loss = 0.0
    current_acc = 0.0
    model.eval()

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            log_ps = model(inputs)
            current_loss += criterion(log_ps, labels)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            current_acc += torch.mean(equals.type(torch.FloatTensor))

    total_loss = current_loss / len(test_dataloader)
    total_acc = current_acc / len(test_dataloader)
    return total_loss, total_acc


def save_checkpoint(model, file_path, arch):
    checkpoint = {'classifier': model.classifier,
                  'state_dict': model.classifier.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'arch': arch}
    torch.save(checkpoint, file_path)


if __name__ == '__main__':
    print('Parsing arguments...')
    args = get_arguments()
    print('Arguments are: %s' % str(vars(args)))
    print('Preparing directories...')
    train_dir, valid_dir, test_dir = get_directories(args.data_dir)
    print('Directories are: %s, %s, %s' % (train_dir, valid_dir, test_dir))
    print('Creating data loaders...')
    dataloaders = create_dataloaders(train_dir, valid_dir, test_dir)
    print('Preparing model...')
    model = build_model(args.arch, args.hidden_units)
    device = 'cuda' if args.gpu else 'cpu'
    print('Training model on [%s]...' % device)
    train_loss, criterion = train_model(model, dataloaders[0], dataloaders[1], args.learning_rate, args.epochs, device)
    print('Validating model on test dataset...')
    val_loss, val_acc = validate_model(model, criterion, dataloaders[2], device)
    print('Training Loss: {:.3f}; Validation Loss: {:.3f}; Validation Accuracy: {:.3f}'.format(train_loss, val_loss,
                                                                                               val_acc))
    print('Saving model to checkpoint...')
    save_checkpoint(model, args.save_dir, args.arch)
