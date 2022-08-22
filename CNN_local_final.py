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
import pickle
from PIL import Image
from PIL import ImageFile





products_df = '/Users/paddy/Desktop/AiCore/facebook_ml/final_dataset/combined_final_dataset.csv'
image_folder = '/Users/paddy/Desktop/AiCore/facebook_ml/images_for_combined/'
batch_size = 32

class ImagesLoader(torch.utils.data.Dataset):

    def __init__(self, transform: transforms = None, labels_level : int=0):
        self.products = pd.read_csv(products_df, lineterminator='\n')
        self.root_dir = image_folder
        self.transform = transform
        self.products['category'] = self.products['category'].apply(lambda x: self.get_category(x, labels_level))
        self.image_id = self.products['image_id']
        self.labels = self.products['category'].to_list()
        self.num_classes = len(set(self.labels))


        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}

        

        if transform == None:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(128),
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
                ])
        assert len(self.labels) == len(self.image_id)
    
        """
        The function takes in a dataframe of products, a folder of images, a transform, a labels_level
        (which is the level of the category tree you want to use), and a max_desc_len (which is the
        maximum length of the description you want to use). 
        
        The function then creates a list of labels, a list of descriptions, a list of image_ids, and a
        number of classes. 
        
        It then creates an encoder and decoder for the labels. 
        
        It then creates a transform if one is not provided. 
        
        It then creates a tokenizer and a vocabulary.
        
        :param transform: This is the transformation that will be applied to the image
        :type transform: transforms
        :param labels_level: This is the level of the labels you want to use. For example, if you want
        to use the top level labels, you would set this to 0. If you want to use the second level
        labels, you would set this to 1, defaults to 0
        :type labels_level: int (optional)
        :param max_desc_len: The maximum length of the description. If the description is longer than
        this, it will be truncated, defaults to 50 (optional)
        """
        



    def __len__(self):
        return len(self.products)

        """
        The function returns the length of the products list
        :return: The length of the products list.
        """


    def __getitem__(self, index):
        """
        The function takes in an index, and returns the image, description, and label of the product at
        that index
        
        :param index: the index of the image in the dataset
        :return: The image, the description, and the label.
        """
        label = self.labels[index]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        image = Image.open(self.root_dir + (self.products.iloc[index, 1] + '.jpg')).convert('RGB')
        image = self.transform(image)

        return image, label



    @staticmethod
    def get_category(x, level: int = 0):
        return x.split('/')[level].strip()



dataset = ImagesLoader()

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
    def __init__(self, decoder: dict = None):
        """
        We're taking the pretrained ResNet50 model, freezing the first 47 layers, and then adding a few
        more layers to the end of the model
        """
        super(CNN, self).__init__()
        self.features = models.resnet50(pretrained=True).to(device)
        self.decoder = decoder
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
            torch.nn.Linear(128, 13)
            )


    def forward(self, x):
        """
        It takes in an image, applies the convolutional layers, and then flattens the output of the
        convolutional layers into a vector
        
        :param x: the input to the model
        :return: The output of the forward pass of the model.
        """
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        return x


def get_default_device():
    """
    It returns the device object representing the default device type
    :return: The device object
    """
    """Picking GPU if available or else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()


model = CNN()
model.to(device)


def train_model(model, epochs):
    """
    We train the model for a number of epochs, and for each epoch we iterate through the training and
    validation samples, and for each sample we calculate the loss and accuracy, and then we update the
    model parameters
    
    :param model: the model we want to train
    :param epochs: number of epochs to train for
    """
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


def check_accuracy(loader, model):
    """
    The function takes in a data loader and a model, and checks the accuracy of the model on the data in
    the loader
    
    :param loader: the data loader
    :param model: A PyTorch Module giving the model to train
    """
    model.eval()
    if loader == train_samples:
        print('Checking accuracy on training set')
    else:
        print('Checking accuracy on evaluation set')
    num_correct = 0
    num_samples = 0
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

        
    model_save_name = 'image_cnn.pt'
    path = f"/Users/paddy/Desktop/AiCore/facebook_ml_final/{model_save_name}" 
    torch.save(model.state_dict(), path)
    with open('image_decoder.pkl', 'wb') as f:
            pickle.dump(dataset.decoder, f)
        

if '__name__" == __main__':
    train_model(model, 10)
    model_save_name = 'image_cnn.pt'
    path = f"/Users/paddy/Desktop/AiCore/facebook_ml_final/{model_save_name}" 
    torch.save(model.state_dict(), path)
    with open('image_decoder.pkl', 'wb') as f:
            pickle.dump(dataset.decoder, f)
    check_accuracy(train_samples, model)
    check_accuracy(val_samples, model)






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



# image_test = ImageProcessor()
# var = image_test(Image.open('/Users/paddy/Desktop/AiCore/facebook_ml/Images/0a1baaa8-4556-4e07-a486-599c05cce76c.jpg'))
# print(var.shape)








# %%
