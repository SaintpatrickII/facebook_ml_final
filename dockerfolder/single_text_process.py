

# # %%



#%%
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from transformers import BertModel
import pandas as pd



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
        
        # description = description.unsqueeze(0)

        return description
    '''
    __getitem__:
    
    overwrites __getitem__ magic method, required to be able to index items in the dataset
    '''


if __name__ == '__main__':
    text_test = TextProcessor()
    var = text_test('big ole shelf sdc frvdf erfsdc efvsdc qdfergv')
    print(var)
    print(var.shape)

    # %%