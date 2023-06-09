# -*- coding: utf-8 -*-
"""Copy of Copy of HooHacks.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Mfy5A_eFnvMTnvqVPtDhyb4GR0rApEKx
"""

# !rmdir /content/data/test/".ipynb_checkpoints"
# !rmdir /content/data/train/".ipynb_checkpoints"
# !rmdir /content/data/".ipynb_checkpoints"

import matplotlib.pyplot as plt
import numpy as np
from torch.optim import lr_scheduler
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import zipfile
from pathlib import Path
import os
import shutil
import math
from torchvision import datasets, transforms
import torchvision.models as models
import random

img_dataset = zipfile.ZipFile("dataset-resized (4).zip", 'r')
img_dataset.extractall()
img_dataset.close()

img_dataset = zipfile.ZipFile("rawimgs.zip", 'r')
img_dataset.extractall()
img_dataset.close()

path = Path(os.getcwd())/"data"
path

def get_labels(waste_type, indexes):
  name = [waste_type+str(i)+".jpg" for i in indexes]
  return name

waste_types = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# for waste in waste_types:
#   folder = os.path.join('data', 'train', waste)
#   if not os.path.exists('train'):
#     os.makedirs('train')

#   folder = os.path.join('data', 'test', waste)
#   if not os.path.exists('test'):
#     os.makedirs('test')

# if not os.path.exists(os.path.join('data','test')):
#     os.makedirs(os.path.join('data','test'))

def move_files(source, dest):
  for file in source:
    print(file, dest)
    shutil.move(file, dest)

def moveSecondDataset():
  dataset_classifications = ["AluCan", "Glass", "HDPEM", "PET"]
  for classification in dataset_classifications:
    lst = os.listdir(Path(os.getcwd())/f"rawimgs/{classification}")
    number_files = len(lst)
    print(number_files)
    # train_source = []
    # test_source = []
    dest_class = ""

    if classification == "AluCan":
      dest_class = "metal"
    elif classification == "Glass":
      dest_class = "glass"
    else:  
      dest_class = "plastic"

    for i in range(number_files):
      if i < math.floor(0.8*int(number_files)):
        #print("train", os.path.join(org_path, 'dataset-resized', waste, waste+str(i+1)+".jpg"))
        #train_source.append(os.path.join(org_path, 'dataset-resized', waste, waste+str(i+1)+".jpg"))
        random_file = random.choice(lst)
        print(random_file)
        reg_path = Path(os.getcwd())/f"rawimgs/{classification}"
        full_path = f"{reg_path}/{random_file}"
        try:
          shutil.move(full_path, os.path.join(path, 'train', dest_class) )
        except:
          pass
      else:
        #print("test", os.path.join(org_path, 'dataset-resized', waste, waste+str(i+1)+".jpg"))
        #test_source.append(os.path.join(org_path, 'dataset-resized', waste, waste+str(i+1)+".jpg"))
        random_file = random.choice(lst)
        print(random_file)
        Path(os.getcwd())/f"rawimgs/{classification}"
        full_path = f"{reg_path}/{random_file}"
        try:
          shutil.move(full_path, os.path.join(path, 'test', dest_class) )
        except:
          pass

#   print(train_source)
# #  test_source = [os.path.join(src, waste+str(idx)+".jpg") for idx in range(math.floor(0.2*int(number_files)), 1)]

#   print(test_source)
  
#   for file in train_source:
#     try:
#       shutil.move(file,os.path.join(path, 'train', waste) )
#     except:
#       pass

#   for file in test_source:
#     try:
#       shutil.move(file,os.path.join(path, 'test', waste) )
#     except:
#       pass

for waste in waste_types:
  org_path = Path(os.getcwd())
  src = os.path.join('dataset-resized', waste)
  print(os.path.join(org_path, 'dataset-resized', waste))

  lst = os.listdir(Path(os.getcwd())/f"dataset-resized/{waste}")
  number_files = len(lst)
  print(number_files)
  train_source = []
  test_source = []
  
  for i in range(number_files):
    if i < math.floor(0.8*int(number_files)):
      #print("train", os.path.join(org_path, 'dataset-resized', waste, waste+str(i+1)+".jpg"))
      #train_source.append(os.path.join(org_path, 'dataset-resized', waste, waste+str(i+1)+".jpg"))
      random_file = random.choice(lst)
      #print(random_file)
      reg_path = Path(os.getcwd())/f"dataset-resized/{waste}"
      full_path = f"{reg_path}/{random_file}"
      try:
        shutil.move(full_path, os.path.join(path, 'train', waste) )
      except:
        pass
    else:
      #print("test", os.path.join(org_path, 'dataset-resized', waste, waste+str(i+1)+".jpg"))
      #test_source.append(os.path.join(org_path, 'dataset-resized', waste, waste+str(i+1)+".jpg"))
      random_file = random.choice(lst)
      #print(random_file)
      reg_path = Path(os.getcwd())/f"dataset-resized/{waste}"
      full_path = f"{reg_path}/{random_file}"
      try:
        shutil.move(full_path, os.path.join(path, 'test', waste) )
      except:
        pass

#   print(train_source)
# #  test_source = [os.path.join(src, waste+str(idx)+".jpg") for idx in range(math.floor(0.2*int(number_files)), 1)]

#   print(test_source)
  
#   for file in train_source:
#     try:
#       shutil.move(file,os.path.join(path, 'train', waste) )
#     except:
#       pass

#   for file in test_source:
#     try:
#       shutil.move(file,os.path.join(path, 'test', waste) )
#     except:
#       pass

moveSecondDataset()

transform = transforms.Compose(
    [
     transforms.Resize((224,224)),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
    ]
)

weights = torchvision.models.ResNet18_Weights.DEFAULT
auto_transforms = weights.transforms()

# train_data = datasets.ImageFolder(
#     root = os.path.join(path, 'train'),
#     transform = transform
# )

# test_data = datasets.ImageFolder(
#     root = os.path.join(path, 'test'),
#     transform = transform
# )

train_data = datasets.ImageFolder(
    root = os.path.join(path, 'train'),
    transform = auto_transforms
)

test_data = datasets.ImageFolder(
    root = os.path.join(path, 'test'),
    transform = auto_transforms
)

batch_size = 32
num_workers = os.cpu_count()
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size = batch_size,
    num_workers=num_workers,
    shuffle=True,
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size = batch_size,
    num_workers=num_workers,
    shuffle=True,
)



def imshow(img):
  #img = img / 2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1,2,0)))
  plt.show()

dataiter = iter(train_dataloader)
images, labels = next(dataiter)

print()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%s' % waste_types[labels[j]] for j in range(batch_size)))

# class CNNModel(nn.Module):
#   def __init__(self):
#     super(CNNModel, self).__init__()
#     self.conv1 = nn.Conv2d(3,6,5)
#     self.pool = nn.MaxPool2d(2,2)
#     self.conv2 = nn.Conv2d(6, 16, 5)
#     self.fc1 = nn.Linear(256*256, 512)
#     self.fc2 = nn.Linear(512, 512)
#     self.fc3 = nn.Linear(512, 6)

#   def forward(self, x):
#     x = self.pool(F.relu(self.conv1(x)))
#     x = self.pool(F.relu(self.conv2(x)))
#     x = x.flatten()
#     x = F.relu(self.fc1(x))
#     x = F.relu(self.fc2(x))
#     x = self.fc3(x)
#     return x
  
# net = CNNModel()
# print(net)

class AlexNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.network = models.resnet18(weights=weights)
    last_layer = self.network.fc.in_features
    self.network.fc = nn.Linear(last_layer, 6)
  
  def forward(self, x):
    return self.network(x)
  
net = AlexNet()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)

epoch = 6
total_loss = 0
for h in range(epoch):
  running_loss = 0.0
  for i, data in enumerate(train_dataloader, 0):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    total_loss += loss.item()
    if i % 2000 == 0:
      print(f"{running_loss / 2000}, {h}")
      running_loss = 0.0
print(total_loss / 2000)

# save
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
# reload
net = AlexNet()
net.load_state_dict(torch.load(PATH))

# net = AlexNet()
# net.load_state_dict(torch.load("./cifar_net (1).pth"))

dataiter_test = iter(test_dataloader)
images_test, images_labels = next(dataiter_test)

outputs = net(images_test)

_, predicted = torch.max(outputs, 1)

imshow(torchvision.utils.make_grid(images_test))
print('GroundTruth: ', ' '.join('%s' % waste_types[images_labels[j]] for j in range(batch_size)))
print(predicted.size())
print('Predicted: ', ' '.join('%s' % waste_types[predicted[j]]
                              for j in range(batch_size)))

correct = 0
total = 0
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

y_pred = []
y_true = []

# iterate over test data
for inputs, labels in test_dataloader:
        output = net(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

# constant for classes
classes = ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash')

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')