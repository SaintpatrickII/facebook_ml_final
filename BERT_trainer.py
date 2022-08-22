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

class productsPreProcessing(Dataset):
    def __init__(self, labels_level: int = 0, max_length: int= 100):
        super().__init__()
        self.df = pd.read_csv(products)
        products_df = self.df
        products_df['category'] = products_df['category'].apply(lambda x: self.get_category(x, labels_level))
        self.labels = products_df['category'].to_list()
        self.descriptions = products_df['product_description']
        self.num_classes = len(set(self.labels))
        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.model.eval()
        self.max_length = max_length
        

        """
        This function takes in a dataframe of products, and returns a list of labels and a list of
        descriptions. 
        The labels are the categories of the products, and the descriptions are the product
        descriptions. 
        The function also creates a dictionary that maps the labels to integers, and another dictionary
        that maps the integers to labels. 

        - creates a tokenizer that will be used to tokenize the product descriptions. 
        - creates a BERT model that will be used to encode the product descriptions. 
        - sets the maximum length of the product descriptions. 
        - sets the number of classes, which is the number of unique labels. 
        - sets the labels_level, which is the level of the labels. 
        - sets the max_length, which is the maximum length of the product descriptions. 
        - sets the dataframe


        :param labels_level: This is the level of the category you want to use. For example, if you want
        to use the category "Clothing, Shoes & Jewelry", you would set this to 0. If you want to use the
        category "Women", you would set this to 1, defaults to 0
        :type labels_level: int (optional)
        :param max_length: The maximum length of the input sequence, defaults to 100
        :type max_length: int (optional)
        """
        
    

   
    @staticmethod
    def get_category(x, level: int = 0):
        return x.split('/')[level].strip()

    """
    It takes a string, splits it on the forward slash character, and returns the item at the
    specified index
    
    :param x: the string to be split
    :param level: The level of the category to return. For example, if the category is
    "Books/Non-Fiction/Science", then level 0 is "Books", level 1 is "Non-Fiction", and level 2 is
    "Science", defaults to 0
    :type level: int (optional)
    :return: The category of the product.
    """




    def __len__(self):
        return len(self.descriptions)

    '''
    __len__:
    
    overwrites len from Dataset Abstract class
    '''


    def __getitem__(self, idx):
        label = self.labels[idx]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        sentence = self.descriptions[idx]
        encoded = self.tokenizer.batch_encode_plus([sentence], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
        
        description = description.squeeze(0)

        return description, label


    '''
    __getitem__:
    
    overwrites __getitem__ magic method, required to be able to index items in the dataset
    '''


# %%


dataset = productsPreProcessing()

class CNN(torch.nn.Module):
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
                                    nn.Linear(384 , 128),
                                    nn.ReLU(),
                                    nn.Linear(128, num_classes))
        self.decoder = decoder
    

    """
        The function takes in an input size, number of classes, and a decoder dictionary. It then
        creates a sequential model.
        
        :param input_size: The size of the input to the model. This is the size of the output of the
        BERT model, defaults to 768
        :type input_size: int (optional)
        :param num_classes: number of classes in the dataset, defaults to 13
        :type num_classes: int (optional)
        :param decoder: dict = None
        :type decoder: dict
        """
    
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


# sets dataloader variables
validation_split = 0.15
batch_size =32
shuffle_dataset = True
random_seed = 42


dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]


# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_samples = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler, drop_last=True)
val_samples = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler, drop_last=True)

def train_model(model, epochs):
    writer = SummaryWriter()
    model.train()
    print('training model')
    optimiser = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        for phase in [train_samples, val_samples]:
            if phase == train_samples:
                print('training')
                model.train()
            else:
                model.eval()
                print('val')

            for i, (features, labels) in tqdm(enumerate(phase)):
                if phase == 'train':
                    torch.set_grad_enabled(phase)
                num_correct = 0
                num_samples = 0
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

                if i % 130 == 129:
                    print(f'Epoch {epoch + 1}/{epochs}')
                    print('-' * 10)
                    if phase == train_samples:
                        writer.add_scalar('Training Loss', loss, epoch)
                        writer.add_scalar(' Training Accuracy', acc, epoch)
                        print('training_loss')
                        print(f'Loss: {loss:.4f} Acc: {acc*100:.1f}%')
                        print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')
                    else:
                        writer.add_scalar('Validation Loss', loss, epoch)
                        writer.add_scalar('Validation Accuracy', acc, epoch)
                        print('val_loss') 
                        print(f'Loss: {loss:.4f} Acc: {acc*100:.1f}%')
                        print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')
                        writer.flush()
                

    """
    train model:

    We're training the model, and for each epoch, we're iterating through the training and validation
    samples, and for each mini-batch, we're calculating the loss and accuracy, and then we're
    backpropagating the loss and updating the weights. 

    We're also using a SummaryWriter to write the loss and accuracy to TensorBoard.

    :param model: the model we want to train
    :param epochs: number of times to iterate over the entire dataset
    """


def check_accuracy(loader, model):
    model.eval()
    if loader == train_samples:
        print('Checking accuracy on training set')
    else:
        print('Checking accuracy on evaluation set')
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for feature, label in loader:
            scores = model(feature)
            _, preds = scores.max(1)
            num_correct += (preds == label).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')

    model_save_name = 'text_cnn.pt'
    path = f"/Users/paddy/Desktop/AiCore/facebook_ml_final/{model_save_name}" 
    torch.save(model.state_dict(), path)
    with open('text_decoder.pkl', 'wb') as f:
        pickle.dump(dataset.decoder, f)
   
   
    """
    check accuracy:

    It takes a data loader and a model, and checks the accuracy of the model on the data in the loader
    for both training & validation sets
    :param loader: the data loader for the dataset to check
    :param model: A PyTorch Module giving the model to train
    """

if '__name__" == __main__':
    cnn = CNN()
    print(dataset[0])
    train_model(cnn, 10)
    model_save_name = 'text_cnn.pt'
    path = f"/Users/paddy/Desktop/AiCore/facebook_ml_final/{model_save_name}" 
    torch.save(cnn.state_dict(), path)
    with open('text_decoder.pkl', 'wb') as f:
        pickle.dump(dataset.decoder, f)
    check_accuracy(train_samples, cnn)
    check_accuracy(val_samples, cnn)



# %%