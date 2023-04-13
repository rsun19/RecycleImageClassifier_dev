from flask import Flask, request, render_template
from imageApp import app
import torch
import numpy as np
import requests
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
from PIL import Image
from torchvision import datasets, transforms
import torchvision.models as models

BASE = "https://127.0.0.1:5000/"

weights = torchvision.models.ResNet18_Weights.DEFAULT
auto_transforms = weights.transforms()

class AlexNet(nn.Module):
  def __init__(self):
    super(AlexNet, self).__init__()
    self.network = models.resnet18(weights=weights).eval()
    last_layer = self.network.fc.in_features
    self.network.fc = nn.Linear(last_layer, 6)
  
  def forward(self, x):
    return self.network(x)
  
  
@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

model = AlexNet()
model.load_state_dict(torch.load("imageApp/cifar_net (4).pth"))
#model = torch.jit.load('imageApp/cifar_net.pth')

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

@app.route("/results", methods=['POST'])
def results():
    # Get uploaded image file
    image = request.files['image']

    if image.filename.endswith(".heic"):
        image.__format__ = 'jpg'

    # Process image and make prediction
    image_tensor = process_image(Image.open(image))
    output = model(image_tensor)

    # Get class probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)
    probabilities = probabilities.detach().numpy()[0]

    # Get the index of the highest probability
    class_index = probabilities.argmax()

    # Get the predicted class and probability
    predicted_class = waste_types[class_index]
    predicted_class = str(predicted_class).capitalize()
    probability = probabilities[class_index]

    # Sort class probabilities in descending order
    class_probs = list(zip(waste_types, probabilities))
    class_probs.sort(key=lambda x: x[1], reverse=True)

    # Render HTML page with prediction results
    return render_template('results.html', class_probs=class_probs,
                           predicted_class=predicted_class, probability=probability*100)

def process_image(image):
    transformation = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    image = image.resize((384, 512))
    #image = image.resize((512, 384))
    #image_tensor = transformation(image)
    image_tensor = auto_transforms(image).unsqueeze(0)
    #image_tensor = weights.transforms(image)
    #image_tensor = transformation(image).unsqueeze(0)
    return image_tensor

waste_types = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


