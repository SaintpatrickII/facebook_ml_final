#%%
import torch
from torch import Tensor
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as transforms





class ImagesLoader(Dataset):
    def __init__(self, json_file='/Users/paddy/Desktop/AiCore/facebook_ml/final_dataset/image_data.json', root_dir='/Users/paddy/Desktop/AiCore/facebook_ml/Images', transform=None):
        self.read_json = pd.read_json(json_file)
        self.root_dir = root_dir
        self.transform = transform
        assert len(json_file[0]) == len(json_file[1])

    def __len__(self):
        return len(self.read_json)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.read_json.iloc[index, 0])
        features = io.imread(f'{img_path}.jpg')
        features = torch.tensor(features).float()
        features = features.reshape(3, 128, 128)
        labels = torch.tensor(self.read_json.iloc[index, 1])
        features = features/255
        if self.transform:
            features = self.transform(features)

        return (features, labels)

        

transform = transforms.Compose([
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        
    ),
    transforms.RandomHorizontalFlip(),

])




dataset = ImagesLoader(transform=transform)

train_split = 0.7
validation_split = 0.15
batch_size = 32

data_size = len(dataset)
print(f'dataset contains {data_size} Images')

train_size = int(train_split * data_size)
val_size = int(validation_split * data_size)
test_size = data_size - (val_size + train_size)
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_samples = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_samples = DataLoader(val_data, batch_size=batch_size)
test_samples = DataLoader(test_data, batch_size=batch_size)









class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = models.resnet50(pretrained=True).to(device)
        for i, param in enumerate(self.features.parameters()):
            if i < 47:
                param.requires_grad=False
            else:
                param.requires_grad=True
        self.features.fc = nn.Sequential(
            nn.Linear(2048, 1024), # first arg is the size of the flattened output from resnet50
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear((128), 13)
            )


    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        return x


def get_default_device():
    """Picking GPU if available or else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()


model = CNN()
model.to(device)


def train_model(model, epochs):
    writer = SummaryWriter()
    model.train()
    print('training model')
    optimiser = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        for phase in [train_samples, val_samples]:
            if phase == train_samples:
                print('training')
            else:
                print('val')
            for i, (features, labels) in enumerate(phase):
                num_correct = 0
                num_samples = 0
                features, labels = features, labels
                features = features.to(device)  # move to device
                labels = labels.to(device)
                predict = model(features)
                labels = labels
                loss = F.cross_entropy(predict, labels)
                _, preds = predict.max(1)
                num_correct += (preds == labels).sum()
                num_samples += preds.size(0)
                acc = float(num_correct) / num_samples
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
                # writer.add_scalar('Loss', loss, epoch)
                # writer.add_scalar('Accuracy', acc, epoch)
                if i % 10 == 9:
                    if phase == train_samples:
                      writer.add_scalar('Training Loss', loss, epoch)
                      writer.add_scalar(' Training Accuracy', acc, epoch)
                      print('training_loss')
                    else:
                      writer.add_scalar('Validation Loss', loss, epoch)
                      writer.add_scalar('Validation Accuracy', acc, epoch)
                      print('val_loss') 
                    # print(batch) # print every 50 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss}')
                    print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')
                    writer.flush()
            

train_model(model, 50)


def check_accuracy(loader, model):
    model.eval()
    if loader == train_samples:
        # model.train()
        print('Checking accuracy on training set')
    else:
        print('Checking accuracy on evaluation set')
        # model.eval()
    num_correct = 0
    num_samples = 0
    #   tells model not to compute gradients
    with torch.no_grad():
        for feature, label in loader:
            feature = feature.to(device)  # move to device
            label = label.to(device)
            scores = model(feature)
            _, preds = scores.max(1)
            num_correct += (preds == label).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')
      
        



check_accuracy(train_samples, model)
check_accuracy(val_samples, model)



model_save_name = 'cnn.pt'
path = f"/Users/paddy/Desktop/AiCore/facebook_ml/{model_save_name}" 
torch.save(model.state_dict(), path)
<<<<<<< HEAD






# %%


import torchvision.transforms as transforms
from PIL import Image
class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]) # is this right?
            ])

        self.transform_Gray = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Lambda(self.repeat_channel),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def repeat_channel(x):
            return x.repeat(3, 1, 1)

    def __call__(self, image):
        if image.mode != 'RGB':
            image = self.transform_Gray(image)
        else:
            image = self.transform(image)
        
        # Add a dimension to the image
        image = image[None, :, :, :]
        return image



image_test = ImageProcessor()
var = image_test(Image.open('/Users/paddy/Desktop/AiCore/facebook_ml/Images/0a1baaa8-4556-4e07-a486-599c05cce76c.jpg'))
print(var.shape)








# %%
=======
>>>>>>> f2ad08c734372b267d827f1e778359060c38f46b
