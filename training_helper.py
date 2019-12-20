import os
import numpy as np
from tqdm import trange
import torch
from torch import utils
from torchvision import transforms, datasets

from early_stopping import EarlyStopping


def get_loaders():
    data_dir = 'dogImages/'
    train_dir = os.path.join(data_dir, 'train/')
    valid_dir = os.path.join(data_dir, 'valid/')
    test_dir = os.path.join(data_dir, 'test/')
    debug_dir = os.path.join(data_dir, 'debug/')

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
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    test_data = datasets.ImageFolder(debug_dir, transform=data_transforms['test'])

    batch_size = 1
    num_workers = 0

    train_loader = utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

    return loaders


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path, scheduler, patience=15):
    """returns trained model"""
    early_stopping = EarlyStopping(patience=patience)
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in trange(1, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # find the loss and update the model parameters accordingly
            # record the average training loss, using something like

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to the model
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # update the average validation loss
            # forward pass to get net output
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update validation loss
            valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        scheduler.step(valid_loss)
        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss
        ))

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load last checkpoint with the best model
    model.load_state_dict(torch.load('saved_models/checkpoint.pt'))
    best_vloss = -early_stopping.best_score
    torch.save(model.state_dict(), f'{save_path}_vloss{best_vloss:.5f}.pt')
    print('Finished Training')
    # return trained model
    return model


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
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))