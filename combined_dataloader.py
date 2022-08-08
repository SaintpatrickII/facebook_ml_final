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

products_df = '/Users/paddy/Desktop/AiCore/facebook_ml/final_dataset/combined_final_dataset.csv'
image_folder = '/Users/paddy/Desktop/AiCore/facebook_ml/images_for_combined/'
batch_size = 32

class ImageTextDataloader(torch.utils.data.Dataset):

    def __init__(self, transform: transforms = None, labels_level : int=0, max_desc_len = 50):
        
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
        self.max_desc_len = max_desc_len
        self.products['category'] = self.products['category'].apply(lambda x: self.get_category(x, labels_level))
        self.descriptions = self.products['product_description']
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


        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = self.get_vocab()
        assert len(self.descriptions) == len(self.labels) == len(self.image_id)
    

    def get_vocab(self):
        
        """
        We use the tokenizer to tokenize each description in the dataset, and then we use the
        build_vocab_from_iterator function to build the vocabulary from the tokenized descriptions
        :return: A dictionary of the words in the vocab and their index.
        """

        def yield_tokens():
            for description in self.descriptions:
                tokens = self.tokenizer(description)
                yield tokens
        token_generator = yield_tokens()

        vocab = build_vocab_from_iterator(token_generator, specials=['<UNK>'])
        print('length of vocab:', len(vocab))
        return vocab


    def tokenize_descriptions(self, descriptions):
        
        """
        The function takes in a list of descriptions, and returns a list of tokenized descriptions
        
        :param descriptions: a pandas series of descriptions
        :return: A list of tokenized descriptions
        """
        def tokenize_description(description):
            words = self.tokenizer(description)
            words = words[:50]
            pad_length = self.max_desc_len - len(words)
            words.extend(['<UNK>'] * pad_length)
            tokenized_desc = self.vocab(words)
            tokenized_desc = torch.tensor(tokenized_desc)
            return tokenized_desc

        descriptions = tokenize_description(descriptions)
        # .apply(tokenize_description)
        return descriptions


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
        encoded = self.tokenize_descriptions(sentence)
        description = encoded
        return image, description, label




    @staticmethod
    def get_category(x, level: int = 0):
        return x.split('/')[level].strip()


if __name__ == '__main__':
    dataset = ImageTextDataloader()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=12,shuffle=True, num_workers=1)
    
    # print(dataset[3000])
    # print('-'*10)
    # print(dataset.decoder[int(dataset[3000][2])])
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=12,
    #                                          shuffle=True, num_workers=1)
    for i, (image, description, labels) in enumerate(dataloader):
        print(image)
        print(description)
        print(labels)
        # print(description.size())
        print(image.size())
        if i == 0:
            break
# /Users/paddy/Desktop/AiCore/facebook_ml/Images/0a3267c5-b660-4bef-9915-56b11cd67d8b.jpg
#%%
