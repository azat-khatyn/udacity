## Description of the project

Creating and training an image classifier to recognize different species of flowers.

PyTorch was used to train a deep learning network based on a VGG pretrained network.
The classifier consists of one layer with 1024 neurons and a ReLu activation function followed by the output layer with 102 neurons and a Softmax activation function.

Description of the files:
- `Image Classification Project.ipynb` - Jupyter notebook with the whole code;
- `train.py` contains code to train the network and to save it to the checkpoint;
- `predict.py`contains the application code, that loads from the checkpoint and then predicts.

This project can be run on CPU or on GPU if available.
