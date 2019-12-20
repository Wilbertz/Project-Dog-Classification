import os
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, datasets
import torch.nn as nn
from PIL import Image, ImageFile
from training_helper import train, test, get_loaders

ImageFile.LOAD_TRUNCATED_IMAGES = True

use_cuda = torch.cuda.is_available()

data_dir = 'dogImages/'
train_dir = os.path.join(data_dir, 'train/')
valid_dir = os.path.join(data_dir, 'valid/')
test_dir = os.path.join(data_dir, 'test')

train_data = datasets.ImageFolder(train_dir)
class_names = [item[4:].replace("_", " ") for item in train_data.classes]

criterion_transfer = nn.CrossEntropyLoss()

model_transfer = models.resnet18(pretrained=True)

for param in model_transfer.parameters():  # model_transfer.features.parameters() for vgg16
    param.requires_grad = False

model_transfer.fc = nn.Linear(model_transfer.fc.in_features, 133)

model_transfer.load_state_dict(torch.load('saved_models/model_transfer_cpu.pt', map_location='cpu'))


def predict_breed_transfer_old(img_path):
    model_transfer.eval()
    # load the image and return the predicted breed
    img = Image.open(img_path).convert('RGB')
    size = (224, 224)  # ResNet image size requirements
    transform_chain = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    img = transform_chain(img).float().unsqueeze(0)

    if use_cuda:
        img = img.cuda()

    model_out = model_transfer(img)
    pred = model_out.data.max(1, keepdim=True)[1]

    if use_cuda:
        model_out = model_out.cpu()

    prediction = torch.argmax(model_out)

    return class_names[prediction]  # predicted class label


def predict_breed_transfer(img_path):
    model_transfer.eval()

    loader = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                                                      (0.229, 0.224, 0.225))])
    image = Image.open(img_path)
    image = loader(image).float()
    # needs to  be Variable to be accepted by model
    image = Variable(image)
    # makes a mini-batch of size 1
    image = image.unsqueeze(0)
    # get predictions, squeeze it out of the 'mini-batch', and return as numpy
    prediction = model_transfer(image).squeeze().data.numpy()
    # np.argmax returns the position of the largest value
    predicted_label_idx = np.argmax(prediction)
    return class_names[predicted_label_idx]  # predicted class label


loaders_transfer = get_loaders()


def test(loaders, model, criterion, use_cuda):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        pred2 = target.data

        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)

result = predict_breed_transfer('dogImages/test/004.Akita/Akita_00244.jpg')
print(result)

print('Done')
