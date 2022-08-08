import json
import glob
import pandas as pd
import numpy as np
import sklearn
from numpy import asarray
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torch import flatten
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = '/Users/paddy/Desktop/AiCore/facebook_ml/final_dataset/image_data.json'
images = '/Users/paddy/Desktop/AiCore/facebook_ml/images_raw'
images_corrected = '/Users/paddy/Desktop/AiCore/facebook_ml/Images'

df = pd.read_json(data)

class image_setup():
    
    def __init__(self):
        image_list = glob.glob(f'{images}/*.jpg')
        df = pd.read_json(data)
        self.image_id =  df['image_id'].to_numpy()
        self.image_array = []
        self.image_classification = df['category'].to_numpy()    
        self.list_of_tuples = []

        np.vectorize(self.image_correct_prop())
        self.image_to_tensor()
        self.joiner()
        self.logreg()


    def image_correct_prop(self):
        resized_img = 128
        for img in range(len(self.image_id)):
            image_ite = Image.open(f'{images}/{self.image_id[img]}.jpg')
            background = Image.new(mode='RGB', size=(resized_img, resized_img)) 
            original_img = image_ite.size
            max_dim = max(image_ite.size)
            ratio = resized_img / max_dim
            img_corr_ratio =(int(original_img[0] * ratio), int(original_img[1] * ratio))
            img_corr = image_ite.resize(img_corr_ratio)
            background.paste(img_corr, (((resized_img - img_corr_ratio[0]) // 2), ((resized_img - img_corr_ratio[1]) // 2)))
            background.save(f'{images_corrected}/{self.image_id[img]}.jpg')
        return print('images have been resized')


    def image_to_tensor(self):
        for img in self.image_id:
            image = Image.open(f'{images_corrected}/{img}.jpg')
            print(image)
            t = ToTensor()
            tensor = t(image)
            flattened = torch.flatten(tensor)
            flatten_numpy = flattened.numpy()
            self.image_array.append(flatten_numpy)
            print(flatten_numpy)
        print('images have been turned into numpy arrays')
        return self.image_array



    def joiner(self):
        for i in range(len(self.image_array)):
            classification = self.image_classification[i]
            image_tuple = self.image_array[i], classification
            self.list_of_tuples.append(image_tuple)
        print('images are now in tuple form')
        return 


    def logreg(self):
        m = len(self.list_of_tuples)
        array_size = 49152
        self.list_of_tuples
        print(len(self.list_of_tuples))
        X = np.zeros((m, array_size))
        y = np.zeros(m)
        print(X.shape, y.shape)
        for item in range(m):
            features, labels = self.list_of_tuples[item]
            X[item, :] = features
            y[item] = labels
        model = LogisticRegression(max_iter= 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print ('accuracy:', accuracy_score(y_test, pred))
        report = ('report:', classification_report(y_test, pred))
        print('here is the report :)', report)
        return report

image_setup()
   