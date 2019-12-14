
import os
import re
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.autograd import Variable
from PIL import ImageFile

import squeeze_net

ImageFile.LOAD_TRUNCATED_IMAGES = True
data_dir = 'dogImages/'
train_dir = os.path.join(data_dir, 'train/')
data_transforms = {
        'train': transforms.Compose([
            transforms.RandomAffine(10, translate=[0.1, 0.1], shear=10),
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4, ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((230, 230)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((230, 230)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
class_names = [item[4:].replace("_", " ") for item in train_data.classes]

model_transfer = squeeze_net.Squeezenet()
model_transfer.load_state_dict(torch.load('saved_models/model_transfer.pt_vloss0.82785.pt', map_location='cpu'))


def predict_breed_transfer(img_path):
    # load the image and return the predicted breed
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

    correct_classification = False
    true_label_idx = -1
    if re.search(r"\d+", img_path) != None:
        true_label_idx = int(re.search(r"\d+", img_path).group(0)) - 1

    return class_names[np.argmax(prediction)], predicted_label_idx, true_label_idx


print(predict_breed_transfer('dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg'))
