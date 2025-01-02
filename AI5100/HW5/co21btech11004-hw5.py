#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split

import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset


# In[ ]:


# seed 
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


# ## Data
# - Replicated transformation from resnet paper
# - We follow the simple data augmentation in [24] for training: 4 pixels are padded on each side,
# and a 32×32 crop is randomly sampled from the padded
# image or its horizontal flip. For testing, we only evaluate
# the single view of the original 32×32 image.

# In[2]:


# load cifar dataset from data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5, test_batch


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data():
    data = []
    labels = []
    for i in range(1, 6):
        batch = unpickle(f'./cifar10/cifar-10-batches-py/data_batch_{i}')
        data.append(batch[b'data'])
        labels += batch[b'labels']
    batch = unpickle('./cifar10/cifar-10-batches-py/test_batch')
    data.append(batch[b'data'])
    labels += batch[b'labels']
    data = np.concatenate(data)
    data = data.reshape(data.shape[0], 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))
    return data, labels

data, labels = load_cifar10_data()
print(data.shape)


# In[3]:


# load cifar10 dataset
trainTransform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainTransform_alex = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testTransform_alex = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# In[ ]:


# split data into train, validation and test with 70%, 10%, 20%, make trainloader, validloader, testloader

train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2, random_state=42)
train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label, test_size=0.125, random_state=42)

# aplly trainTransform to train_data and testTransform to test_data, valid_data


class CifarDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            # convert numpy array to PIL image
            image = image.astype(np.uint8)
            image = transforms.ToPILImage()(image)
            image = self.transform(image)
        return image, label

def get_train_val_test(batch_size, trainTransform, testTransform):

    train_dataset = CifarDataset(train_data, train_label, trainTransform)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    valid_dataset = CifarDataset(valid_data, valid_label, testTransform)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    test_dataset = CifarDataset(test_data, test_label, testTransform)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return trainloader, validloader, testloader


# ## Train Function

# In[5]:


def trainFunc(model, trainloader, optimizer, criteria, device, log_file):
    # return loss, accuracy, and number of correct predictions, for each batch
    model.train()
    train_loss = []
    train_acc = []
    for i, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criteria(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        train_acc.append(correct)
        # print accuracy and loss for each batch
        # print('Train: Batch: %d, Loss: %.3f, Accuracy: %.3f' % (i, loss.item(), correct / len(targets)))
        log_file.write('Train: Batch: %d, Loss: %.3f, Accuracy: %.3f\n' % (i, loss.item(), correct / len(targets)) )
    return train_loss, train_acc


# ## Test Function

# In[6]:


def testFunc(model, testloader, criteria, device):
    # return loss, accuracy, and number of correct predictions, for each batch
    model.eval()
    total_loss = 0
    total_correct = 0
    total_data = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criteria(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_data += targets.size(0)
    # print accuracy and loss for each batch
    return total_loss / len(testloader), total_correct / total_data


# In[ ]:


# train and test model, save train_loss, train _acc for each bach, test_loss, test_acc for each epoch as log file
def trainModel(model, trainloader, validloader, testloader, optimizer, criteria, scheduler, device, res_dir, exp_name, EPOCHS):
    log_file = open(f'{res_dir}/{exp_name}/log.txt', 'w')

    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    test_loss = []
    test_acc = []

    # save best model
    best_acc = 0
    bestAcc_epoch = 0
    best_model = None

    # start time
    start_time = time.time()
    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch}')
        log_file.write(f'Epoch: {epoch}\n')
        train_loss_, train_acc_ = trainFunc(model, trainloader, optimizer, criteria, device, log_file)
        scheduler.step()
        train_loss.extend(train_loss_)
        train_acc.extend(train_acc_)
        valid_loss_, valid_acc_ = testFunc(model, validloader, criteria, device)
        valid_loss.append(valid_loss_)
        valid_acc.append(valid_acc_)
        test_loss_, test_acc_ = testFunc(model, testloader, criteria, device)
        test_loss.append(test_loss_)
        test_acc.append(test_acc_)

        if test_acc_ > best_acc:
            best_acc = test_acc_
            bestAcc_epoch = epoch
            best_model = model.state_dict()
            torch.save(best_model, f'{res_dir}/{exp_name}/best_model.pth')

        # print epoch, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc
        print(f'Epoch: {epoch}, Train Loss: {np.mean(train_loss_)}, Train Accuracy: {np.sum(train_acc_) / len(train_data)}, Validation Loss: {valid_loss_}, Validation Accuracy: {valid_acc_}, Test Loss: {test_loss_}, Test Accuracy: {test_acc_}')
        log_file.write(f'Epoch: {epoch}, Train Loss: {np.mean(train_loss_)}, Train Accuracy: {np.sum(train_acc_) / len(train_data)}, Validation Loss: {valid_loss_}, Validation Accuracy: {valid_acc_}, Test Loss: {test_loss_}, Test Accuracy: {test_acc_}\n')

    # save model
    torch.save(model.state_dict(), f'{res_dir}/{exp_name}/model.pth')

    log_file.write(f'Best Accuracy: {best_acc}, Epoch: {bestAcc_epoch}\n')
    
    # end time
    end_time = time.time()

    log_file.write(f'Training Time: {end_time - start_time}\n')

    log_file.close()


# ## Resnet50 Training

# In[9]:


BATCH_SIZE = 256
EPOCHS = 300


# In[10]:


# load resnet50 model from hub
resnet_model = torch.hub.load('pytorch/vision', 'resnet50', weights=None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_model.to(device)

model_name = 'resnet50_t'

trainloader, validloader, testloader = get_train_val_test(BATCH_SIZE, trainTransform, testTransform)

# print the numer of samples in train, validation, test
print('Train size:', len(trainloader.dataset))
print('Validation size:', len(validloader.dataset))
print('Test size:', len(testloader.dataset))


# ## Loss function, Optimizers, Schedulers
# - Refrence from resnet paper
# - We use a weight decay of 0.0001 and momentum of 0.9,
# and adopt the weight initialization in [13] and BN [16] but
# with no dropout.
# - https://www.kaggle.com/code/greatcodes/pytorch-cnn-resnet18-cifar10

# In[11]:


LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
BASE_LR = 0.0001
MAX_LR = 0.1
STEP_SIZE_UP = 2000
CYCLE_MOMENTUM = False

opt_name = 'SGD'
scheduler_name = 'CyclicLR'

criteria = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=BASE_LR, max_lr=MAX_LR, step_size_up=STEP_SIZE_UP, cycle_momentum=CYCLE_MOMENTUM)


# In[12]:


# make resuts directory and save configuration
res_dir = './results'
exp_name = f'{model_name}_{opt_name}_{scheduler_name}'
os.makedirs(f'{res_dir}/{exp_name}', exist_ok=True)

# save training configuration as json
config = {
    'model': model_name,
    'batch_size': BATCH_SIZE,
    'epochs': EPOCHS,
    'optimizer': opt_name,
    'learning_rate': LEARNING_RATE,
    'momentum': MOMENTUM,
    'weight_decay': WEIGHT_DECAY,
    'scheduler': scheduler_name,
    'base_lr': BASE_LR,
    'max_lr': MAX_LR,
    'step_size_up': STEP_SIZE_UP,
    'cycle_momentum': CYCLE_MOMENTUM,
}

with open(f'{res_dir}/{exp_name}/config.json', 'w') as f:
    json.dump(config, f)


# In[12]:


trainModel(resnet_model, trainloader, validloader, testloader, optimizer, criteria, scheduler, device, res_dir, exp_name, EPOCHS)


# ## AlexNet Training

# In[13]:


# load alexnet model from hub
alexnet_model = torch.hub.load('pytorch/vision', 'alexnet', weights=None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
alexnet_model.to(device)

model_name = 'alexnet_t'

trainloader, validloader, testloader = get_train_val_test(BATCH_SIZE, trainTransform_alex, testTransform_alex)

# print the numer of samples in train, validation, test
print('Train size:', len(trainloader.dataset))
print('Validation size:', len(validloader.dataset))
print('Test size:', len(testloader.dataset))


# In[14]:


# using the same config for loss, optimizer, scheduler
criteria = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(alexnet_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=BASE_LR, max_lr=MAX_LR, step_size_up=STEP_SIZE_UP, cycle_momentum=CYCLE_MOMENTUM)


# In[15]:


# make resuts directory and save configuration
res_dir = './results'
exp_name = f'{model_name}_{opt_name}_{scheduler_name}'
os.makedirs(f'{res_dir}/{exp_name}', exist_ok=True)

# save training configuration as json
config = {
    'model': model_name,
    'batch_size': BATCH_SIZE,
    'epochs': EPOCHS,
    'optimizer': opt_name,
    'learning_rate': LEARNING_RATE,
    'momentum': MOMENTUM,
    'weight_decay': WEIGHT_DECAY,
    'scheduler': scheduler_name,
    'base_lr': BASE_LR,
    'max_lr': MAX_LR,
    'step_size_up': STEP_SIZE_UP,
    'cycle_momentum': CYCLE_MOMENTUM,
}

with open(f'{res_dir}/{exp_name}/config.json', 'w') as f:
    json.dump(config, f)


# In[16]:


trainModel(alexnet_model, trainloader, validloader, testloader, optimizer, criteria, scheduler, device, res_dir, exp_name, EPOCHS)

