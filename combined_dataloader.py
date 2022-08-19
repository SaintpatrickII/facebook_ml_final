#%%
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
from torch import Tensor
import torch.nn as nn 
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import io
from PIL import Image
from PIL import ImageFile
from transformers import BertTokenizer
from transformers import BertModel

products_df = '/Users/paddy/Desktop/AiCore/facebook_ml/final_dataset/combined_final_dataset.csv'
image_folder = '/Users/paddy/Desktop/AiCore/facebook_ml/images_for_combined/'
batch_size = 32

class ImageTextDataloader(torch.utils.data.Dataset):

    def __init__(self, transform: transforms = None, labels_level : int=0, max_len = 100):
        
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
        self.products = pd.read_csv(products_df, lineterminator='\n')
        self.root_dir = image_folder
        self.transform = transform
        self.products['category'] = self.products['category'].apply(lambda x: self.get_category(x, labels_level))
        self.descriptions = self.products['product_description']
        self.image_id = self.products['image_id']
        self.labels = self.products['category'].to_list()
        self.num_classes = len(set(self.labels))


        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.model.eval()
        self.max_length = max_len
        

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


        # self.tokenizer = get_tokenizer('basic_english')
        assert len(self.descriptions) == len(self.labels) == len(self.image_id)
    



    def __len__(self):
        """
        The function returns the length of the products list
        :return: The length of the products list.
        """
        return len(self.products)


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
        sentence = self.descriptions[index]
        encoded = self.tokenizer.batch_encode_plus([sentence], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
        
        description = description.squeeze(0)
        return image, description, label




    @staticmethod
    def get_category(x, level: int = 0):
        return x.split('/')[level].strip()


if __name__ == '__main__':
    dataset = ImageTextDataloader()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=True, num_workers=1)
    
    # print(dataset[3000])
    # print('-'*10)
    # print(dataset.decoder[int(dataset[3000][2])])
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=12,
    #                                          shuffle=True, num_workers=1)
    for i, (image, description, labels) in enumerate(dataloader):
        print(image)
        print(description)
        print(labels)
        print(description.size())
        print(image.size())
        print(labels.size())
        if i == 0:
            break
# /Users/paddy/Desktop/AiCore/facebook_ml/Images/0a3267c5-b660-4bef-9915-56b11cd67d8b.jpg
#%%
