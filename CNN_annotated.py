# %%
from pyexpat import features
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pytorch_loader import ImagesLoader
import numpy as np
import matplotlib.pyplot as plt
plt.ion() 
from time import sleep

# torch imported, nn is our neural network import, optim is for optimisation algos i.e. grad desc, functional gives us relu activation function, transforms to transform data from numpy to torch tensors
#  dataloader allows for easy minibatching, imagesloader loads our dataset

#  creating the neural network
# class inherits from the torch nn package, by calling super we initilise the nn package from the parent class, here we set parameters of the model, for this example we have 10188 images/nodes & 13 classes

dataset = ImagesLoader()






train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
for batch in train_loader:
    print(batch)
    break





def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy().reshape(128, 128, 3)
    
    plt.imshow(npimg)
    # np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# sleep(50)



# %%



# for the initiliser we have input size which is the number of images & the number of classes, this init inherits from the pytorch nn module
class CNN(nn.Module):
    def __init__(self, input_size=10188, no_classes=13):
        super(CNN, self).__init__()
# self.layers is our CNN iterator, in this we pass our inputs through convulutional layers, layers can be stacked on each other to increase complexity of model, as layers increase its a good idea to pool
# layers as
        self.layers = torch.nn.Sequential(
            
            torch.nn.Conv2d(3, 8, 11),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(11, 3),
            torch.nn.Conv2d(8, 32, 9),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(11, 3),
            torch.nn.Flatten(),
            torch.nn.Linear(1152, 13),
            # torch.nn.Linear(576, 288),
            #  torch.nn.Linear(288, 13),
            torch.nn.Softmax()
        )




    def forward(self, features):
        # x = f.relu(self.layers(x))
        # return x
        return self.layers(features)
    # relu calls non linearity function on every element, here all -ve values are removed in attempt to provide more complex learning
        

'''


# model = CNN().float()
# optimiser = optim.SGD(model.parameters(), lr = 0.01, momentum=0.7)
# writer = SummaryWriter()

# def train_model():
#     for epoch in range(50):
#         for batch_idx, batch in enumerate(train_loader):
            
#             features, labels = batch
#             prediction = model(features)
#             loss = F.cross_entropy(prediction, labels)
#             # print(prediction)
#             # print('Batch no:', batch_idx, 'Loss:', loss.item())
#             # print(prediction.shape)
#             loss.backward()
#             optimiser.step()
#             optimiser.zero_grad()
#             running_loss = 0
#             writer.add_scalar('Loss', loss, batch_idx)
#             if batch_idx % 500 == 499:    # print every 50 mini-batches
#                 print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {loss}')
#                 running_loss = 0.0

# train_model()
# writer.flush()

'''




# # here we would put model = NeuralNetwork(input, class_size) if we had not specified befor in the init
# model = NeuralNetwork(10188, 13)
# #  the 64 is our mini batch size
# x = torch.randn(64, 10188)
# print(model(x).shape)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# here we would put model = NeuralNetwork(input, class_size) if we had not specified befor in the init
# model = CNN(10188, 13)
#  the 64 is our mini batch size
# x = torch.rand n(64, 10188)
# print(model(x).shape)

# -------------- above is a test, our x shape should be the batch size & amount of classes



