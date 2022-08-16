# # %%
# from torch.utils.data.sampler import SubsetRandomSampler
# # import pandas as pd
# import numpy as np
# import torch
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator
# from torch.utils.data import DataLoader, Dataset
# import torch.nn.functional as F
# import torch
# from torch import Tensor
# import torch.nn as nn 
# import torch.optim as optim
# import torchvision.transforms as transforms
# from torchvision import models, datasets
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader
# import os
# import torch
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms
# from skimage import io
# from PIL import Image
# from PIL import ImageFile

# products_df = '/Users/paddy/Desktop/AiCore/facebook_ml/final_dataset/combined_final_dataset.csv'
# image_folder = '/Users/paddy/Desktop/AiCore/facebook_ml/images_for_combined/'

# class TextProcessor(torch.utils.data.Dataset):

#     def __init__(self, max_desc_len = 100):
#         """
#         The function takes in a maximum description length and returns a tokenizer
        
#         :param max_desc_len: The maximum length of the description. If the description is longer than
#         this, it will be truncated, defaults to 50 (optional)
#         """
#         self.max_desc_len = max_desc_len
#         self.tokenizer = get_tokenizer('basic_english')
#         # self.vocab = self.get_vocab()

    

#     def get_vocab(self, text):
#         """
#         The function takes in a text and tokenizes it. Then it builds a vocabulary from the tokenized
#         text
        
#         :param text: the text to be tokenized
#         :return: A dictionary of the vocab
#         """

#         def yield_tokens():
#             tokens = self.tokenizer(text)
#             yield tokens
#         token_generator = yield_tokens()

#         vocab = build_vocab_from_iterator(token_generator, specials=['<UNK>'])
#         print('length of vocab:', len(vocab))
#         return vocab


#     def get_vocab_length(self, vocab):
#         # vocab = self.vocab
#         vocab_len = len(vocab)
#         print(vocab_len)
#         return vocab_len

#     def tokenize_descriptions(self, descriptions):
#         """
#         The function takes in a list of descriptions, and returns a list of tokenized descriptions
        
#         :param descriptions: a pandas series of descriptions
#         :return: A list of tokenized descriptions
#         """
#         def tokenize_description(description):
#             words = self.tokenizer(description)
#             words = words[:100]
#             pad_length = self.max_desc_len - len(words)
#             words.extend(['<UNK>'] * pad_length)
#             tokenized_desc = self.vocab(words)
#             tokenized_desc = torch.tensor(tokenized_desc)
#             return tokenized_desc

#         descriptions = tokenize_description(descriptions)
#         # .apply(tokenize_description)
#         return descriptions

#     def __call__(self, text):
#         """
#         The function takes in a sentence, tokenizes it, and returns the tokenized sentence
        
#         :param text: The text to be tokenized
#         :return: The encoded description of the sentence.
#         """
#         self.vocab = self.get_vocab(text)
#         self.vocab_len = self.get_vocab_length(text)
#         sentence = text
#         encoded = self.tokenize_descriptions(sentence)
#         description = encoded
#         return description




#     @staticmethod
#     def get_category(x, level: int = 0):
#         return x.split('/')[level].strip()


# if __name__ == '__main__':
#     text_test = TextProcessor()
#     var = text_test('big ole shelf, sdc, frvdf, erfsdc, efvsdc, qdfergv')
#     print(var)
#     print(var.size)


# # text_test = TextProcessor()
# # var = text_test('big ole shelf, sdc, frvdf, erfsdc, efvsdc, qdfergv')
# # print(var)
# # print(var.size)


# # %%



#%%
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import pickle
from transformers import BertTokenizer
from transformers import BertModel
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import pickle
import torch.nn as nn
from tqdm import tqdm


products = '/Users/paddy/Desktop/AiCore/facebook_ml/final_dataset/combined_final_dataset.csv'

df =  pd.read_csv(products)
df.head

class TextProcessor(Dataset):
    def __init__(self, labels_level: int = 0, max_length: int= 100):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.model.eval()
        self.max_length = max_length
        
        
    

   
    @staticmethod
    def get_category(x, level: int = 0):
        return x.split('/')[level].strip()

    def __len__(self):
        return len(self.descriptions)


    '''
    __len__:
    
    overwrites len from Dataset Abstract class
    '''


    def __call__(self, text):
        sentence = text
        encoded = self.tokenizer.batch_encode_plus([sentence], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
        
        description = description.squeeze(0)

        return description
    '''
    __getitem__:
    
    overwrites __getitem__ magic method, required to be able to index items in the dataset
    '''


if __name__ == '__main__':
    text_test = TextProcessor()
    var = text_test('big ole shelf sdc frvdf erfsdc efvsdc qdfergv')
    print(var)
    print(var.size)

    # %%