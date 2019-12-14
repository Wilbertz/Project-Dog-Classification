import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from PIL import ImageFile

import Squeezenet
from training_helper import train, test, get_loaders

ImageFile.LOAD_TRUNCATED_IMAGES = True

use_cuda = torch.cuda.is_available()

loaders_transfer = get_loaders()

# Specify model architecture
model_transfer = Squeezenet.Squeezenet()

if use_cuda:
    model_transfer = model_transfer.cuda()

criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.Adam(params=model_transfer.classifier.parameters())
# reduce learning rate when a validation loss has stopped improving
plateau_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_transfer, 'min',  patience=7, verbose=True)


model_transfer = train(500, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda,
                       'saved_models/model_transfer.pt', plateau_lr_scheduler)

model_transfer.load_state_dict(torch.load('saved_models/model_transfer.pt_vloss0.82785.pt', map_location='cpu'))

test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)
