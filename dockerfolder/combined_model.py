# %%
from tqdm import tqdm
from combined_dataloader import ImageTextDataloader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import numpy as np
from torch import Tensor
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from pytorch_loader import ImagesLoader
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as transforms
import pickle


products_df = '/Users/paddy/Desktop/AiCore/facebook_ml/final_dataset/combined_final_dataset.csv'
image_folder = '/Users/paddy/Desktop/AiCore/facebook_ml/images_for_combined/'


# validation_split = 0.15
# batch_size = 32
# shuffle_dataset = True
# random_seed = 42

# dataset = ImageTextDataloader(Image_dir=image_folder, csv_file=products_df, transform=None)
# dataset_size = len(dataset)
# print(dataset[4000])
# # print(dataset_size)
# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# if shuffle_dataset :
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]

# # Creating PT data samplers and loaders:
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)

# train_samples = torch.utils.data.DataLoader(ImageTextDataloader, batch_size=batch_size, 
#                                            sampler=train_sampler)
# val_samples = torch.utils.data.DataLoader(ImageTextDataloader, batch_size=batch_size,
#                                                 sampler=valid_sampler)


# %%

def get_default_device():
    """
    It checks if GPU is available and returns the device accordingly.
    :return: The device is being returned.
    """
    """Picking GPU if available or else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()




class TextClassifier(torch.nn.Module):
    def __init__(self,
                 input_size: int = 768,
                 num_classes: int = 13,
                 decoder: dict = None):
        super().__init__()
        self.main = nn.Sequential(nn.Conv1d(input_size, 256, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=2, stride=2),
                                    nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(384, 128)).to(device)
        self.decoder = decoder
    def forward(self, inp):
        x = self.main(inp)
        return x

    """
    forward: 

    The function takes in a batch of sentences, passes them through the embedding layer, and then
    passes them through the layers of the model
    :param X: the input data
    :return: The output of the last layer of the network.
    """


class ImageTextClassifier(nn.Module):
    def __init__(self):
        """
        We're taking the pretrained resnet50 model, freezing the first 47 layers, and then adding a new
        fully connected layer to the end of the model
        """
        super(ImageTextClassifier, self).__init__()
        self.features = models.resnet50(pretrained=True).to(device)
        self.text_model = TextClassifier()
        self.main = nn.Sequential(nn.Linear(256, 13))
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
            torch.nn.Linear(512, 128)
            # torch.nn.ReLU(),
            # torch.nn.Linear((128), 13)
            )


    def forward(self, image_features, text_features):
        """
        We take the image features, pass them through the image model, flatten them, pass the text
        features through the text model, concatenate the two, and pass the result through the main model
        
        :param image_features: The output of the image model
        :param text_features: The text features that we extracted from the text
        :return: The combined features of the image and text features.
        """
        image_features = self.features(image_features)
        image_features = image_features.reshape(image_features.shape[0], -1)
        text_features = self.text_model(text_features)
        combined_features = torch.cat((image_features, text_features), 1)
        combined_features = self.main(combined_features)
        return combined_features

 
model = ImageTextClassifier()
model.to(device)


dataset = ImageTextDataloader()
dataloader = dataloader = torch.utils.data.DataLoader(dataset, batch_size=32 ,shuffle=True, num_workers=1)
# print(dataset[5000])



def train_model(model, epochs):
    """
    We are training the model by iterating through the dataloader, and for each batch we are calculating
    the loss and accuracy, and then updating the model parameters
    
    :param model: the model we want to train
    :param epochs: number of epochs to train for
    """
# optimiser):
# scheduler
    writer = SummaryWriter()
    print('training model')
    # dataset_ite = tqdm(enumerate(dataloader))
    optimiser = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        for i, (image_features, text_features, labels) in tqdm(enumerate(dataloader)):
            model.train()
            num_correct = 0
            num_samples = 0
            image_features = image_features.to(device)
            text_features = text_features.to(device)  # move to device
            labels = labels.to(device)
            predict = model(image_features, text_features)
            labels = labels


            loss = F.cross_entropy(predict, labels)
            _, preds = predict.max(1)
            num_correct += (preds == labels).sum()
            num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()


            if i % 130 == 129:
                break
                writer.add_scalar('Training Loss', loss, epoch)
                writer.add_scalar(' Training Accuracy', acc, epoch)
                print('training_loss')
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss:.5f}')
                print(f'Got {num_correct} / {num_samples} with accuracy: {(acc * 100):.2f}%')
                writer.flush()





def check_accuracy(loader, model):
    """
    We iterate through the training set, and for each image and text pair, we pass it through the model
    and get a prediction. 
    
    We then compare the prediction to the actual label, and if they match, we increment the number of
    correct predictions. 
    
    At the end, we calculate the accuracy by dividing the number of correct predictions by the total
    number of predictions
    
    :param loader: the data loader
    :param model: The model to train
    """
    model.eval()
    print('Checking accuracy on training set')
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for (image_features, text_features, label) in tqdm(loader):
            image_features = image_features.to(device)
            text_features = text_features.to(device)  # move to device
            label = label.to(device)
            predict = model(image_features, text_features)
            _, preds = predict.max(1)
            num_correct += (preds == label).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')
        


    model_save_name = 'combined_final.pt'
    path = f"/Users/paddy/Desktop/AiCore/facebook_ml_final/{model_save_name}" 
    torch.save(model.state_dict(), path)
    with open('combined_decoder_final.pkl', 'wb') as f:
        pickle.dump(dataset.decoder, f)
                    
if __name__ == '__main__':
    train_model(model, 10)

    model_save_name = 'combined.pt'
    path = f"/Users/paddy/Desktop/AiCore/facebook_ml_final/{model_save_name}" 
    torch.save(model.state_dict(), path)
    with open('combined_decoder.pkl', 'wb') as f:
        pickle.dump(dataset.decoder, f)

    check_accuracy(dataloader, model)
                    
# if __name__ == '__main__':
#     train_model(model, 10)
#     # optimiser_ft)
#     check_accuracy(dataloader, model)
#     # model_save_name = 'combined.pt'
#     # path = f"/Users/paddy/Desktop/AiCore/facebook_ml/{model_save_name}" 

# %%



