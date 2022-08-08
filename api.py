import fastapi
<<<<<<< HEAD

=======
<<<<<<< HEAD
>>>>>>> f2ad08c734372b267d827f1e778359060c38f46b
from fastapi import FastAPI, File
from fastapi import Request
from fastapi import UploadFile
from fastapi import Form
<<<<<<< HEAD
from grpc import StatusCode
=======
>>>>>>> f2ad08c734372b267d827f1e778359060c38f46b
import uvicorn
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from single_image_process import ImageProcessor
<<<<<<< HEAD
from single_text_process import TextProcessor
from combined_model import TextClassifier
# , ImageTextClassifier
=======
=======
>>>>>>> f2ad08c734372b267d827f1e778359060c38f46b
from fastapi import FastAPI
from fastapi import Request
from fastapi import UploadFile
import uvicorn
<<<<<<< HEAD
import pickle
import requests
import json
from fastapi.responses import JSONResponse
=======
>>>>>>> 7bc87a140c00ea2e8ae7b6e8a77233468cd39b18
>>>>>>> f2ad08c734372b267d827f1e778359060c38f46b

app = FastAPI()


<<<<<<< HEAD
image_processor = ImageProcessor()
text_processor = TextProcessor()
=======
<<<<<<< HEAD
image_processor = ImageProcessor()
>>>>>>> f2ad08c734372b267d827f1e778359060c38f46b


# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = 'cpu'


<<<<<<< HEAD


class ImageTextClassifier(nn.Module):
    def __init__(self, decoder=None):
        super(ImageTextClassifier, self).__init__()
        self.features = models.resnet50(pretrained=True).to(device)
        self.text_model = TextClassifier()
        self.main = nn.Sequential(nn.Linear(256, 13))
=======
class CNN(nn.Module):
    def __init__(self, num_classes, decoder : dict = None):
        super(CNN, self).__init__()
        self.features = models.resnet50(pretrained=True).to(device)

>>>>>>> f2ad08c734372b267d827f1e778359060c38f46b
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
<<<<<<< HEAD
            torch.nn.Linear(128, 13)
            )
        self.decoder = decoder

    def forward(self, image_features, text_features):
        image_features = self.features(image_features)
        image_features = image_features.reshape(image_features.shape[0], -1)
        text_features = self.text_model(text_features)
        combined_features = torch.cat((image_features, text_features), 1)
        return combined_features


    def predict(self, combined):
        with torch.no_grad():
            x = self.forward(combined)
            return x

    def predict_prob(self, combined):
        with torch.no_grad():
            x = self.forward(combined)
            return torch.softmax(x, dim=1)


    def predict_class(self, combined):
        with torch.no_grad():
            x = self.forward(combined)
            return self.decoder(int(torch.argmax(x, dim=1)))


with open('combined_decoder.pkl', 'rb') as f:
    combined_decoder = pickle.load(f)
image_model = ImageTextClassifier(decoder=combined_decoder)
image_model.load_state_dict(torch.load('combined.pt', map_location='cpu'))



# @app.get('/example')
# def test_get(x):
#     print(x)
#     return 'get was successful'

@app.post('/test')
def test_post(image : UploadFile = File(...), text: str = Form(...)):
    img = Image.open(image.file)

    processed_image = image_processor(img)
    processed_text = text_processor(text)

    prediction = image_model.predict(processed_image, processed_text)
    pred_prob = image_model.predict_prob(processed_image, processed_text)
    class_pred = image_model.predict_class(processed_image, processed_text)
    print(prediction)
    print(pred_prob)
    print(class_pred)
    return JSONResponse(status_code=200, content={'prediction' : prediction.tolist(), 'probability': pred_prob.tolist(), 'class': class_pred})


# @app.post('/text')
# def test_text(text: str = Form(...)):
#     print(text)
#     return 'yyyyeeeessss'

if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port=8080)

# @app.post('/test')
# def test_post(image : UploadFile(...)):
#     print(image)
#     return 'bye'


# if __name__ == '__main__':
#     uvicorn.run('api:app', host='0.0.0.0', port='8080')
=======
            torch.nn.Linear((128), num_classes)
            )
        self.decoder = decoder


    def forward(self, x):
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
            return self.decoder(int(torch.argmax(x, dim=1)))


image_model = CNN(num_classes=13, decoder=imagedecoder )
image_model.load_state_dict(torch.load('cnn.pt', map_location='cpu'))



@app.get('/example')
def test_get(x):
    print(x)
    return 'get was successful'

@app.post('/test')
def test_post(image : UploadFile = File(...)):
    img = Image.open(image.file)
    processed_image = image_processor(img)
    prediction = image_model.predict(processed_image)
    print(prediction)
    return 'bye'


@app.post('/text')
def test_text(text: str = Form(...)):
    print(text)
    return 'yyyyeeeessss'

if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port=8080)
=======

@app.post('/test')
def test_post(image : UploadFile(...)):
    print(image)
    return 'bye'


if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port='8080')
>>>>>>> 7bc87a140c00ea2e8ae7b6e8a77233468cd39b18
>>>>>>> f2ad08c734372b267d827f1e778359060c38f46b
