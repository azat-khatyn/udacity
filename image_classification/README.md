## Description of the project

Creating and training an image classifier to recognize different species of flowers.

I used PyTorch to train a deep learning network based on a VGG pretrained network.
The classifier I defined consists of one layer with 1024 neurons and a ReLu activation function followed by the output layer with 102 neurons and a Softmax activation function. 
I used value 0.2 for the Dropout for the hidden layer.

This is a CLI application:
- File `train.py` contains code to train the network and to save it to the checkpoint.
- File `predict.py`contains the application code, that loads from the checkpoint and then predicts.
- It can be run on CPU and on GPU
