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
        # features = features.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # .astype(int))
        # labels = labels.unsqueeze(0)
        # if self.transform:
        #     features = self.transform(features)
        return (features, labels)

        # if self.transform:
        #     features = self.transform(features)
        #     return(features, y_label)

dataset1 = ImagesLoader()

# print(len(dataset1))
# print(dataset1[356])
# for idx in range(len(dataset1)):
#     features, labels = dataset1[idx]
#     print(features.shape)
#     print(labels.shape)
#     print(features)
#     print(labels)
#     break
# 10189 photos so far need to rerun loader to resize entire dataset