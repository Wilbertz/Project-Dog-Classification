import torch
import torchvision.models as models
from torchvision import transforms, datasets
import torch.nn as nn
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

use_cuda = torch.cuda.is_available()

model_transfer = models.resnet34(pretrained=True)

for param in model_transfer.parameters():  # model_transfer.features.parameters() for vgg16
    param.requires_grad = False

model_transfer.fc = nn.Linear(model_transfer.fc.in_features, 133)

model_transfer.load_state_dict(torch.load('saved_models/checkpoint.pt', map_location='cpu'))


def predict_breed_transfer(img_path):
    # load the image and return the predicted breed
    img = Image.open(img_path).convert('RGB')
    size = (256, 256)  # ResNet image size requirements
    transform_chain = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    img = transform_chain(img).unsqueeze(0)

    if use_cuda:
        img = img.cuda()

    model_out = model_transfer(img)

    if use_cuda:
        model_out = model_out.cpu()

    prediction = torch.argmax(model_out)

    return class_names[prediction]  # predicted class label


print ('Done')
