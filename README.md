# facebook_ml
Machine Learning project aimed at creating a machine learning that recommends products akin to facebook marketplace

1. Exploring & cleaning the datasets:

Before starting any form of EDA it is important to examine the datasets & clean them ready for usage in any analysis/ models. For this project there are two datasets: a raw text 'products' csv file, this contains all information about the product i.e. price, location, description etc and an images folder containing the raw images associated to the 'products' dataset.


2. Cleaning Datasets

Advertisement Dataset:

- Data is loaded in from the S3 bucket into a jupyter notebook (this allows for rapid troubleshooting, once all cleaning has been completed this can be transferred into a python method), pandas was used to turn the dataset into a dataframe.

- Before any cleaning it is a good idea to have a look at the raw file to see any obvious improvements. a combination of this & the use of the pandas df.dtype and df.head gives us a good indication of the next steps to be taken to clean this file.

Cleaning operations:

1. by looking at the datatypes of our rows most are object, in order to save hastle we convert these to str type, this will later allow us to manipulate them later on

2. from looking at the csv file some non-ascii characters have been used in two of the columns (product description & product name), for this we use a simple encode/decode function to rid the df of these

3. our df has a inbuilt index column & a meaningless additional index column, this is dropped from the table, whilst dropping this column i also chose to drop columns with nan data inputs, this meaning our new df will have inputs for each value

4. the price function contains a '£' symbol in each price, to make these values into floats we simply use a str replace & convert the results into float64 datatype

5. product_name includes the location of the item in the header, this information is already known in the location column so we use a str split & index the zeroth element so that product name is only the product

6. category & location both contain two pieces of information, here i have split the category into an additional subcaregory column & with location it is split into county & the original location


Images Dataset:

raw images come in a variety of sizes & aspect ratios, in order to train the machine learning model firstly we will need to clean these images beforehand.Images here are modified through the PIL import & glob is used to iterate through the files within our image folder.

Cleaning Operations:

1. Images are formed in a for loop with an enumerate function(this makes for easy naming of the photos).

2. Firstly our loop opens the image & created a black background at the specified limit of 512 x 512 pixels to be overlayed later.

2. the maximum dimension of each image is found & compared to our maximum acceptable size, this is computed into a ratio factor which will transform the image to the correct size.

3. Background image is overlayed with the product image, image is centred on this background & saved with the enumerate function from before

3. Creating Simple Machine Learning Models:

Product Details Regression:

- A simple model is utiliser here, using one hot encoding products are assigned into their categories paired with their price

- So we are going to be testing the linear dependance category of the price, so y is the price of the item & X is the category

- As this is a very basic test as expected the model doesnt perform well at all, with a MSE of its definetly not ideal

Image Multi Class Classification:

- For the Images again we will be one hot encoding to catogrise products, however we will need to have. numerical value for each category for our images model

- Images are opened alike before using PIL, in this current form our model has no way to analyse the image, for this we will need to transform the image into a readable format

- to do this we can transform the PIL image into a pytorch Tensor, in this form we will have a three channel tensor for each image, this however is still not readable by our model.

- The tensor must be flattened & turned into a flat numpy array for usage

- In this form while trainable is not ideal as we lose massive amounts of information on the images that could be used in a model i.e. do pixels that are neighbors have any effect? This will be fixed at a later point using neural networks

- This flattened numpy array is joined to the class number in a tuple for training in sklearn's logistic regression model with X=(no. of labels, no of features) and y = (no of labels)

- This model only produces an accuracy of 15% this will be drastically increased with the usage of a CNN

- While making this image dataset i had tried for ages to correctly split the categories & index them into the correct tuples, this can be seen in the image_model_final file, for anyone looking to mimic this, for the love of all things good just use the inbuilt sklearn LabelEncoder.

- My other main pause with this step happened when my categories were completeley unbalanced, as it turns out in my data cleaning when removing duplicate rows i had accidently removed rows which contained different pictures of the same product, always good to double check this next time :)


4. Creating a CNN

As we saw beforehand using a simpler ML model like a logistic regression is inefficient for training on an image dataset, in order to train a model for classifying images a convolutional neural network is required. This CNN utilises progressiveley more complex hidden layers to detect patterns in classes of images. Before we can create this CNN the data must be loaded correctly.

- An image loader class is created to load our images, this takes our images from a directory, converts them to tensors, sets up features & labels & introduces a transforms function which we will use to normalise pixel values, both __getitem__ and __len__ methods have to be incuded in the dataloader as required by torch dataloaders

- Images are split into train, validation and test sets in preprocessing

- Resnet-50 is imported as the CNN to be adjusted through transfer learning, in the neural network the first 47 layers repain unfrozen meaning that as our model trains these layers weights will remain untouched.

- The final three layers are unfrozen & additional layers are added to change RESNET-50's output layer of 2048 classes to the 13 in our network. as the model trains during back propergation all these layers will be adjusted to increase accuracy of predictions, a dropout layer is used here also, this is used to help with regularisation

- The forward pass reshapes the tensor output into a one dimensional vector from its original feature map form

- The model is trained through a series of 50 epochs, in each epoch after training we also record validation, taking accuracy and loss of both for each epoch, this is saved in a tensorboard for visulisation

- After 50 epochs a final loss and final accuracy for both training & validation is taken, for this CNN the following results were obtained:

<img width="895" alt="Screenshot 2022-06-07 at 10 59 25" src="https://user-images.githubusercontent.com/92804317/183415570-3100afd5-c85b-4be8-9fa8-7c8dee22b817.png">

<img width="895" alt="Screenshot 2022-06-07 at 10 59 40" src="https://user-images.githubusercontent.com/92804317/183415584-5635842e-1939-4bfb-9e58-330428e7e4c6.png">

<img width="895" alt="Screenshot 2022-06-07 at 11 02 56" src="https://user-images.githubusercontent.com/92804317/183415611-ce410265-2737-4ab4-a4d7-c801afba7c0e.png">

<img width="895" alt="Screenshot 2022-06-07 at 11 03 08" src="https://user-images.githubusercontent.com/92804317/183415647-fe2d082c-0f47-43c1-a479-b54ba5ec110c.png">

5. Creating the Text model

![Screenshot 2022-06-15 at 19 07 42](https://user-images.githubusercontent.com/92804317/183415799-150e6264-4137-4ba4-a19a-0d6bb169bd11.png)

- Alike the image model beforehand the text model must first have a dataloader in order to transform raw text into tensors 

- For this model we will be using the word2vec model in order to create our word embeddings.

- Once the correct products csv file has been loaded in the dataloader can be made, for this there are two key processes that must be made, firstly a 'get_vocab' method, in this method a list of every word in the dataset is recorded, as we train the model different words will be used from the vocab in an attempt to find relationships betweeen words & categories

- the second key methid is to 'tokenize description' here we transform each word into a torch.tensor, every word will have a unique value assigned to it from the vocab method, doing this will allow for the words to be understood by our model

- Dataloader output is a tuple of the torch.tensor of each description & the category of the product as shown below

<img width="895" alt="Screenshot 2022-08-08 at 13 11 25" src="https://user-images.githubusercontent.com/92804317/183415142-3627998c-5881-41de-abee-887f8426ccba.png">

- For training the text model the setup of the CNN works very similar however an aditional Embedding layer is required at the begininning of the model, this is a built in torch.nn function & requires the no. of words in the vocab and the embedding size (size of each sentence vector)

- By training a further 50 epochs wof just the text model the following results were obtained

<img width="895" alt="Screenshot 2022-08-08 at 13 20 06" src="https://user-images.githubusercontent.com/92804317/183417656-df4fb871-fd01-43d0-87af-f9c19da1e3e2.png">


<img width="895" alt="Screenshot 2022-08-08 at 13 21 13" src="https://user-images.githubusercontent.com/92804317/183417296-d9befabc-7bb6-4653-a248-160570d23a04.png">

<img width="895" alt="Screenshot 2022-08-08 at 13 21 21" src="https://user-images.githubusercontent.com/92804317/183417334-ceb61134-260e-4bda-8ac2-092b25868c0e.png">

<img width="895" alt="Screenshot 2022-08-08 at 13 22 07" src="https://user-images.githubusercontent.com/92804317/183417348-7eea7ac1-4b7b-490a-8844-a59987a369a1.png">

<img width="895" alt="Screenshot 2022-08-08 at 13 22 13" src="https://user-images.githubusercontent.com/92804317/183417366-606559c0-a356-413b-bb37-7e9022c790e2.png">

6. Combining models:

- Alike before a dataloader is utilised which completes. the appropriate transformations to both text & images, in this model however we also add an encoder & decoder to the loader, this will translate the class number to its string category name

- Model architecture stays largely the same with one important change, before the final linear layers == no. of classes, now we wat the final layers of each to both == 128, after combining both models one final linear layer is utilised to equate the final output to the number of classes

- Model is trained & state.dict is saved alongside the decoder to be used later within our API

- For the combined mdel, there is a rather large lag before it catches up to that of the other two models as the model is learning off the entire dataset, however within the 10 epochs benchmark given to compare models it hasent reached similar accuracy however doubling the training duration should lead to far greater results ( this took all weekend to train already :) )

<img width="1456" alt="Screenshot 2022-08-22 at 19 29 29" src="https://user-images.githubusercontent.com/92804317/185993688-9540f589-69d6-42bb-b959-fc6cd8a105eb.png">


7. Configure & deploy API call for model:

- API is setup utilising FastAPI, firstly on localhost for ease of testing a .post method is setup using uvicorn to update the localhost whenever the api.py file is ran

- Two external dataloaders are needed as now we will be feeding the model a singular image & text input, so modified loaders are used to account for this instead of the batches previously used, for images we need to add a dimension for the 'batch size' & the text now only tokenizes a single description rather than a list of descriptions

- Text, Image & Combined model is imported & the forward method is called by the API .post method, this will ask for user input of image & text & load -> predict said imputs. a JSONResponse is used to give an output of predictions

- now that this is functional the last step is to host this on the cloud

8. EC2 migration & Docker:

- Before we migrate our api & relevant files to be uploaded to an ec2 a dockerfolder is created to contain all neccasary files to use the api, alongside this a seperate requirements.txt file is created specifically for this folder. 

- Alongside this a docker-compose.yml file is created, this allows for updates of files contained here to update in real time int the docker image and also a dockerfile script which gives commands on how to build the image for our api.

- This dockerfolder is now build using 'docker build tag patrickgovus/test' while in the dockerfolder directory & pushed to dockerhub via 'docker push patrickgovus/ml', to double check once image has been created a simple 'docker images' should now show patrickgovus/ml as an image

- To use this image we simply run 'docker run -p 8080:8080 patrickgovus/ml', the -p flag assigns the local 8080 port to the our virtual docker 8080 port

- As this image will be accepting http requests on docker hub this image will be set as a private repo to avoid any unwanted connections

- As the model .pt files are all over 1gb in size we will have to use a instance larger than the traditional free tier, once this is created we can either just clone the repo (boring) or just get the correct api files by pulling the docker image we just made, within VSCode you can directly ssh intto a instance with extensions so thats exactly what we will do

- Once the instance is created it's quickly updated by running 'sudo apt upgrade' & from here we just login to docker followed by 'sudo docker pull patrickgovus/ml' to download the api image from docker

- Alike before we can simply enter 'docker run -p 8080:8080 patrickgovus/ml' & the image will now be run through the ec2 instance, in this case our localhost uses the ip of the instance rather than our localhost of before meaning that anyone can try the application


<img width="895" alt="Screenshot 2022-08-24 at 19 22 18" src="https://user-images.githubusercontent.com/92804317/186494568-2fc90c0b-2f31-4eba-9e77-827c489db71e.png">
