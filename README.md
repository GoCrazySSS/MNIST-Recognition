# MNIST-Recognition
    A simple convolutional neural network for handwritten number recognition.
## Preparation
### Dataset
    The model is trained and tested on the common MNIST dataset which you can download from [http://yann.lecun.com/exdb/mnist/].
### Environment
    cuda10.0 & cudnn7.4.2 & anaconda3.
    Run install_dependencies.sh, after which you will have the 'data' and 'output' directories. Then, uncompress the MNIST dataset to the 'data' directory.
## Train a new model
    python train.py
    It needs about 4 hours to get the final model 'model_ep_49.pth' in the 'output' directory.
## Test your model
    python test.py
    I got a test accuracy at 0.99459
