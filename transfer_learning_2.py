import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
import torch.nn as nn
from PIL import ImageFile


from training_helper import train, test, get_loaders

ImageFile.LOAD_TRUNCATED_IMAGES = True

use_cuda = torch.cuda.is_available()

loaders_transfer = get_loaders()

# Specify model architecture
model_transfer = models.resnet34(pretrained=True)

for param in model_transfer.parameters():  # model_transfer.features.parameters() for vgg16
    param.requires_grad = False

model_transfer.fc = nn.Linear(model_transfer.fc.in_features, 133)

if use_cuda:
    model_transfer = model_transfer.cuda()

criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.Adam(params=model_transfer.fc.parameters())
# reduce learning rate when a validation loss has stopped improving
plateau_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_transfer, 'min',  patience=7, verbose=True)


model_transfer = train(500, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda,
                       'saved_models/model_transfer.pt', plateau_lr_scheduler)

model_transfer.load_state_dict(torch.load('saved_models/checkpoint.pt', map_location='cpu'))

test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
