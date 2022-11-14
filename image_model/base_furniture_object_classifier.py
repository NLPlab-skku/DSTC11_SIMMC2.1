import os
import json
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm
from torch import nn, optim
from PIL import Image
from torchvision import models
from torch.optim import lr_scheduler
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS = ['train', 'val'] # split only by two.

mean_nums = [0.504, 0.470, 0.438] # [0.485, 0.456, 0.406] -> [0.504, 0.470, 0.438]
std_nums = [0.297, 0.287, 0.274] # [0.229, 0.224, 0.225] -> [0.297, 0.287, 0.274]

train_transform = T.Compose([
  T.ToTensor(),
  T.Resize(size=[256,256]),
  T.Normalize(mean_nums, std_nums)
])
test_transform = T.Compose([
  T.ToTensor(),
  T.Resize(size=[256,256]),
  T.Normalize(mean_nums, std_nums)
])

train_dataset = ImageFolder(root='./data/furniture/251_class_train', transform=train_transform)
test_dataset = ImageFolder(root='./data/furniture/251_class_test', transform=test_transform)

object_class_name = train_dataset.classes

train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = 16, shuffle=False)
dataset_sizes = {"train": len(train_dataset), "test": len(test_dataset)}
data_loaders = {"train": train_loader, "test": test_loader}

model = models.resnext101_32x8d(pretrained=True)
n_features = model.fc.in_features
model.fc = nn.Linear(n_features, len(os.listdir("./data/furniture/251_class_train")))
model.to(device)

def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device,
  scheduler,
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  pbar = tqdm(data_loader)
  for inputs, labels in pbar:
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)

    _, preds = torch.max(outputs, dim=1)

    loss = loss_fn(outputs, labels)

    correct_predictions += torch.sum(preds == labels)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    pbar.set_postfix({'train_loss':np.mean(losses)})

  scheduler.step()

  return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for inputs, labels in data_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)

      outputs = model(inputs)

      preds = outputs
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, labels)

      correct_predictions += torch.sum(preds == labels)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)


def train_model(model, data_loaders, dataset_sizes, device, n_epochs=3):
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  # optimizer = optim.Adam(model.parameters(), lr=0.0001)
  scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
  loss_fn = nn.CrossEntropyLoss()

  best_accuracy = 0

  for epoch in range(n_epochs):

    print(f'Epoch {epoch + 1}/{n_epochs}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
      model,
      data_loaders['train'],    
      loss_fn, 
      optimizer, 
      device, 
      scheduler,
      dataset_sizes['train']
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    
    val_acc, val_loss = eval_model(
      model,
      data_loaders['test'],
      loss_fn,
      device,
      dataset_sizes['test']
    )

    print(f'Test  loss {val_loss} accuracy {val_acc}')
    print()


    if val_acc > best_accuracy:
      torch.save(model.state_dict(), './parameters/object_model_state_funiture.bin')
      best_accuracy = val_acc

  print(f'Best val accuracy: {best_accuracy}')
  
  model.load_state_dict(torch.load('./parameters/object_model_state_funiture.bin'))
  
  return model

# Set model
base_model = model

base_model = train_model(base_model, data_loaders, dataset_sizes, device, n_epochs=40)
