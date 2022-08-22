from email.policy import strict
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




# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = 'cpu'




class ImageTextClassifier(nn.Module):
    def __init__(self, decoder: dict =None):
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
            # torch.nn.Linear(128, 13)
            )
        self.decoder = decoder
        self.main = nn.Sequential(nn.Linear(256, 13))

    def forward(self, image_features, text_features):
        image_features = self.features(image_features)
        image_features = image_features.reshape(image_features.shape[0], -1)
        text_features = self.text_model(text_features)
        combined_features = torch.cat((image_features, text_features), 1)
        combined_features = self.main(combined_features)
        return combined_features


    def predict(self, image, text):
        with torch.no_grad():
            x = self.forward(image, text)
            return x

    def predict_prob(self, image, text):
        with torch.no_grad():
            x = self.forward(image, text)
            return torch.softmax(x, dim=1)


    def predict_class(self, image, text):
        with torch.no_grad():
            x = self.forward(image, text)
            return self.decoder[int(torch.argmax(x, dim=1))]


with open('combined_decoder.pkl', 'rb') as f:
    combined_decoder = pickle.load(f)
combined_model = ImageTextClassifier(decoder=combined_decoder)
combined_model.load_state_dict(torch.load('combined.pt', map_location='cpu'))
# , strict=False)


# @app.get('/example')
# def test_get(x):
#     print(x)
#     return 'get was successful'

@app.post('/testcombined')
def test_post(image : UploadFile = File(...), text: str = Form(...)):
    img = Image.open(image.file)

    processed_image = image_processor(img)
    processed_text = text_processor(text)

    prediction = combined_model.predict(processed_image, processed_text)
    pred_prob = combined_model.predict_prob(processed_image, processed_text)
    class_pred = combined_model.predict_class(processed_image, processed_text)
    print(prediction)
    print(pred_prob)
    print(class_pred)
    return JSONResponse(status_code=200, content={'prediction' : prediction.tolist(), 'probability': pred_prob.tolist(), 'class': class_pred})


# @app.post('/text')
# def test_text(text: str = Form(...)):
#     print(text)
#     return 'yyyyeeeessss'



# vocab_len = text_processor.get_vocab_length()


class TextClassifierSingle(torch.nn.Module):
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
    def forward(self, inp):
        x = self.main(inp)
        return x
    
    
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
            return self.decoder[int(torch.argmax(x, dim=1))]


with open('text_decoder.pkl', 'rb') as f:
    text_decoder = pickle.load(f)

n_classes = len(text_decoder)
print(n_classes)
text_model = TextClassifierSingle(decoder=text_decoder)
#  num_classes= n_classes)
# , vocab_length=vocab_len)
text_model.load_state_dict(torch.load('text_cnn.pt', map_location='cpu'), strict=False)



@app.post('/testtext')
def test_post(text: str = Form(...)):
    processed_text = text_processor(text)
    text_model = TextClassifierSingle(decoder=combined_decoder)
    text_model.load_state_dict(torch.load('text_cnn.pt', map_location='cpu'))
    prediction = text_model.predict(processed_text)
    pred_prob = text_model.predict_prob(processed_text)
    class_pred = text_model.predict_class(processed_text)
    print(prediction)
    print(pred_prob)
    print(class_pred)
    return JSONResponse(status_code=200, content={'prediction' : prediction.tolist(), 'probability': pred_prob.tolist(), 'class': class_pred})






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
    
    
    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

    def predict_prob(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return torch.softmax(x, dim=1)


    def predict_class(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return self.decoder[int(torch.argmax(x, dim=1))]


with open('image_decoder.pkl', 'rb') as f:
    image_decoder = pickle.load(f)

n_classes = len(text_decoder)
print(n_classes)
image_model = CNN(decoder=image_decoder)
#  num_classes= n_classes)
# , vocab_length=vocab_len)
image_model.load_state_dict(torch.load('image_cnn.pt', map_location='cpu'), strict=False)



@app.post('/testimage')
def test_post(image : UploadFile = File(...)):
    img = Image.open(image.file)
    processed_image = image_processor(img)
    image_model_loaded = image_model
    image_model_loaded.load_state_dict(torch.load('image_cnn.pt', map_location='cpu'))
    prediction = image_model_loaded.predict(processed_image)
    pred_prob = image_model_loaded.predict_prob(processed_image)
    class_pred = image_model_loaded.predict_class(processed_image)
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

