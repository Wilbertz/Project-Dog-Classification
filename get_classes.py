import os
from torchvision import transforms, datasets

data_dir = 'dogImages/'
train_dir = os.path.join(data_dir, 'train/')
valid_dir = os.path.join(data_dir, 'valid/')
test_dir = os.path.join(data_dir, 'test/')

train_data = datasets.ImageFolder(train_dir)
class_names = [item[4:].replace("_", " ") for item in train_data.classes]


def get_class_names():
    pass


print(class_names[132])

