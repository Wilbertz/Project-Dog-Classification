
import os
import re
from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
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
        ])
    }
train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
class_names = [item[4:].replace("_", " ") for item in train_data.classes]

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

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
    if re.search(r"\d+", img_path) is not None:
        true_label_idx = int(re.search(r"\d+", img_path).group(0)) - 1

    return class_names[np.argmax(prediction)], predicted_label_idx, true_label_idx


def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def dog_detector(img_path):
    _, imagenet_class, _ = predict_breed_transfer(img_path)

    if 0 <= imagenet_class <= 133:
        return True
    else:
        return False


def get_path_to_breed(breed_idx=0):
    for img_path, label_idx in train_data.imgs:
        if label_idx == breed_idx:
            return img_path


def run_app(img_path):
    img = cv2.imread(img_path)
    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)

    # handle cases for a human face, dog, and neither
    if face_detector(img_path):
        dog_breed, predicted_label_idx, _ = predict_breed_transfer(img_path)
        print("hello, human!")
        plt.show()
        print("You look like a ... \n" + dog_breed)

        img = cv2.imread(get_path_to_breed(predicted_label_idx))
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(cv_rgb)
        plt.show()

    elif dog_detector(img_path):
        dog_breed, predicted_label_idx, true_label_idx = predict_breed_transfer(img_path)

        print(f"hello, dog! Your breed is {class_names[true_label_idx]}")
        plt.show()
        if true_label_idx == predicted_label_idx:
            print("You look like a ... \n" + dog_breed)
        else:
            print("You look like a ... \n" + dog_breed)
            img = cv2.imread(get_path_to_breed(predicted_label_idx))
            cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(cv_rgb)
            plt.show()
    else:
        return "error"
    print("\n-----------------------------------------------------\n")


dog_files = []
human_files = np.array(glob("lfw/*/*"))
for i in range(133):
    dog_files.append(get_path_to_breed(i))

# suggested code, below

# for file in np.hstack((human_files[500:505], dog_files)):
#     run_app(file)

for file in human_files[0:1]:
    run_app(file)

# print(predict_breed_transfer('dogImages/train/001.Affenpinscher/Affenpinscher_00002.jpg'))
