# EVA8 - 2.5 PyTorch101 Assignment

## Problem Statement
Write a neural network that can:
- 1.Takes 2 inputs:
    -An image from the MNIST dataset (say 5), and
    -A random number between 0 and 9, (say 7)
- Gives two outputs:
-the "number" that was represented by the MNIST image (predict 5), and
-the "sum" of this number with the random number and the input image to the network (predict 5 + 7 = 12)

__please refer to the ```question.png``` image from the ```/img``` folder__
![alt text](/imgs/question.png) 
- you can mix fully connected layers and convolution layers
- you can use one-hot encoding to represent the random number input and the "summed" output.
- Random number (7) can be represented as 0 0 0 0 0 0 0 1 0 0
- Sum (13) can be represented as: 
-0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 (one hot encoded representation)
-0b1101 (remember that 4 digits in binary can at max represent 15, so we may need to go for 5 digits. i.e. 10010


## Solution
1. Please Refer to the ```eva8_mnist_random_num_sum.ipynb``` Jupyter notebook for the solution. 
2. However, the Data, NN Architecture, Loss, Optimizers have been discussed below.

### Dataset - MNIST + Random Integers(0-9)

__please refer to the ```mnist.png``` image from the ```/img``` folder__
![Screenshot](/imgs/mnist.png)

#### Inputs
1. MNIST Dataset (Contains the Images and the corresponding labels)

2. ```torch.randint(0, 10)``` API has been used to generate random integers between 0 and 9.
#### Outputs
1. Labels of the corresponding MNIST images for training the Image classifier network.

2. Summation of the true MNIST label and the random integer for the summation part of the network.

### NN Architecture:
The Net2 architecture is used to solve the assignment. Contains the following layers:

    - 7 convolutional layers

    - 2 MaxPooling layers

    - 3 Linear/Fully connected layers

    ```Inputs:```
        __Image__ : 1x28x28 (MNIST Image)

        __RandomNumber__: 0-9
    
    ```Outputs:```
        __label__: Label of the MNIST Image

        __sum_output__: Sum of the predicted label of the MNIST Image and the random number.

### Loss Functions:
##### 1. Loss function of the MNIST image classification:
- Since it is a classification problem, ```nn.CrossEntropyLoss()``` API has been used to compute the cross entropy loss.
##### 2. Loss function of the summation part.
- Ideally, this should ideally be a MSE (Mean square Error) as the summation could be thought of a "regression" problem.
- However, in the problem/assignment statement, we know that the random number input is always bounded between 0-9. And hence the output is also bounded(0-18), and it's going to be an Integer(as it's allowed to use one-hot encoded representation), the entire summation part could be thought of the classification problem.
-Hence, ```nn.CrossEntropyLoss()``` API has been used to compute the cross entropy loss of the one-hot encoded summation output.

__Please refer to the ```epoch_loss.png``` and ```loss_fn.png``` images in the ```/img``` folder__

    ```
    Loss at Epoch 0: 1.3994579280100272
    Loss at Epoch 1: 0.2154729304313439
    Loss at Epoch 2: 0.20129536288048638
    Loss at Epoch 3: 0.027468014342368747
    Loss at Epoch 4: 0.007723255454994563
    Loss at Epoch 5: 0.055382025443283224
    Loss at Epoch 6: 0.07041706720206276
    Loss at Epoch 7: 0.004435703824696713
    Loss at Epoch 8: 0.0015469093507352036
    Loss at Epoch 9: 0.0031506284465735073 
    ```

![alt text here](/imgs/epoch_loss.png)

![alt text here](/imgs/loss_fn.png)


### Optimizer:
```Adam``` Optimizer with learning rate ```lr=0.0001``` has been used as optimizer.


