import fastapi
from fastapi import FastAPI, File
from fastapi import Request
from fastapi import UploadFile
from fastapi import Form
from grpc import StatusCode
import uvicorn
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from single_image_process import ImageProcessor
from single_text_process import TextProcessor
from combined_model import TextClassifier
# , ImageTextClassifier

from fastapi import FastAPI
from fastapi import Request
from fastapi import UploadFile
import uvicorn

import pickle
import requests
import json
from fastapi.responses import JSONResponse


app = FastAPI()



image_processor = ImageProcessor()
text_processor = TextProcessor()

image_processor = ImageProcessor()



# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = 'cpu'




# class ImageTextClassifier(nn.Module):
#     def __init__(self, decoder: dict =None):
#         super(ImageTextClassifier, self).__init__()
#         self.features = models.resnet50(pretrained=True).to(device)
#         self.text_model = TextClassifier()
#         self.main = nn.Sequential(nn.Linear(256, 13))
#         for i, param in enumerate(self.features.parameters()):
#             if i < 47:
#                 param.requires_grad=False
#             else:
#                 param.requires_grad=True
#         self.features.fc = nn.Sequential(
#             nn.Linear(2048, 1024), # first arg is the size of the flattened output from resnet50
#             torch.nn.ReLU(),
#             torch.nn.Dropout(),
#             torch.nn.Linear(1024, 512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, 128)
#             # torch.nn.ReLU(),
#             # torch.nn.Linear(128, 13)
#             )
#         self.decoder = decoder
#         self.main = nn.Sequential(nn.Linear(256, 13))

#     def forward(self, image_features, text_features):
#         image_features = self.features(image_features)
#         image_features = image_features.reshape(image_features.shape[0], -1)
#         text_features = self.text_model(text_features)
#         combined_features = torch.cat((image_features, text_features), 1)
#         combined_features = self.main(combined_features)
#         return combined_features


#     def predict(self, image, text):
#         with torch.no_grad():
#             x = self.forward(image, text)
#             return x

#     def predict_prob(self, image, text):
#         with torch.no_grad():
#             x = self.forward(image, text)
#             return torch.softmax(x, dim=1)


#     def predict_class(self, image, text):
#         with torch.no_grad():
#             x = self.forward(image, text)
#             return self.decoder(int(torch.argmax(x, dim=1)))


# with open('combined_decoder.pkl', 'rb') as f:
#     combined_decoder = pickle.load(f)
# image_model = ImageTextClassifier(decoder=combined_decoder)
# image_model.load_state_dict(torch.load('combined.pt', map_location='cpu'))
# # , strict=False)


# # @app.get('/example')
# # def test_get(x):
# #     print(x)
# #     return 'get was successful'

# @app.post('/test')
# def test_post(image : UploadFile = File(...), text: str = Form(...)):
#     img = Image.open(image.file)

#     processed_image = image_processor(img)
#     processed_text = text_processor(text)

#     prediction = image_model.predict(processed_image, processed_text)
#     pred_prob = image_model.predict_prob(processed_image, processed_text)
#     class_pred = image_model.predict_class(processed_image, processed_text)
#     print(prediction)
#     print(pred_prob)
#     print(class_pred)
#     return JSONResponse(status_code=200, content={'prediction' : prediction.tolist(), 'probability': pred_prob.tolist(), 'class': class_pred})


# @app.post('/text')
# def test_text(text: str = Form(...)):
#     print(text)
#     return 'yyyyeeeessss'





class TextClassifier(torch.nn.Module):
    def __init__(self, pretrained_weights=None, decoder: dict =None):
        """
        We create an embedding layer with 28381 words and 50 dimensions, then we create a sequential
        model with a convolutional layer with 32 filters, a ReLU activation, a convolutional layer with
        64 filters, a dropout layer, a ReLU activation, a flatten layer, and a linear layer with 128
        neurons
        
        :param pretrained_weights: This is the path to the pretrained weights
        """
        super().__init__()
        no_words = 28381
        embedding_size = 100
        self.embedding = torch.nn.Embedding(no_words, embedding_size)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(embedding_size, 32, 2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, 2),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(6272, 13)
        )
        self.decoder = decoder
    
    
    
    
    
    
    def forward(self, X):
        return self.layers(self.embedding(X))
    
    
    def predict(self, text):
        with torch.no_grad():
            x = self.forward(text)
            return x

    def predict_prob(self, text):
        with torch.no_grad():
            x = self.forward(text)
            return torch.softmax(x, dim=1)


    def predict_class(self, text):
        with torch.no_grad():
            x = self.forward(text)
            return self.decoder(int(torch.argmax(x, dim=1)))


with open('text_decoder.pkl', 'rb') as f:
    combined_decoder = pickle.load(f)
text_model = TextClassifier(decoder=combined_decoder)
text_model.load_state_dict(torch.load('text_cnn.pt', map_location='cpu'))

@app.post('/test')
def test_post(text: str = Form(...)):
    # img = Image.open(image.file)

    # processed_image = image_processor(img)
    processed_text = text_processor(text)

    prediction = TextClassifier.predict(processed_text)
    pred_prob = TextClassifier.predict_prob(processed_text)
    class_pred = TextClassifier.predict_class(processed_text)
    print(prediction)
    print(pred_prob)
    print(class_pred)
    return JSONResponse(status_code=200, content={'prediction' : prediction.tolist(), 'probability': pred_prob.tolist(), 'class': class_pred})


if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port=8090)

# @app.post('/test')
# def test_post(image : UploadFile(...)):
#     print(image)
#     return 'bye'


# if __name__ == '__main__':
#     uvicorn.run('api:app', host='0.0.0.0', port='8080')

