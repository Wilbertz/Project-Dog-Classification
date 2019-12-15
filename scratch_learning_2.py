import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import ImageFile

from conv_net import Net
from training_helper import train, test, get_loaders

ImageFile.LOAD_TRUNCATED_IMAGES = True

use_cuda = torch.cuda.is_available()

loaders_scratch = get_loaders()

model_scratch = Net()

if use_cuda:
    model_scratch.cuda()

print(model_scratch)

criterion_scratch = nn.CrossEntropyLoss()

optimizer_scratch = optim.Adam(params=model_scratch.parameters(), lr=0.001)

# reduce learning rate when a validation loss has stopped improving
plateau_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_scratch, 'min',  patience=7, verbose=True)

model_scratch = train(500, loaders_scratch, model_scratch, optimizer_scratch,
                      criterion_scratch, use_cuda, 'saved_models/model_scratch',
                      plateau_lr_scheduler, patience=15)

model_scratch.load_state_dict(torch.load('saved_models/checkpoint_old.pt', map_location='cpu'))

test(loaders_scratch, model_scratch, criterion_scratch, use_cuda)
