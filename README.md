# Heartbeat Sound Segmentation
This project is a machine learning implementation of a regression model that predicts 
the location of the lub and dub in a heartbeat sound. The model is trained on a dataset of 
heartbeat sounds that have been segmented into individual lub and dub sounds.
## Colab
https://colab.research.google.com/drive/1WF_fb7bBC_iWQGQP7MUJJs2qsfLrov98?usp=sharing

## Problem Statement
Heartbeat sounds contain important information about a person's heart health. However, extracting this information can be challenging, 
especially when it comes to identifying the location of the lub and dub sounds within a heartbeat recording. The goal of this project is 
to develop a machine learning model that can accurately predict the location of the lub and dub sounds within a heartbeat recording. By successfully 
identifying these sounds, we can improve the accuracy of heart health diagnosis and monitoring.

The model will be trained on a dataset of segmented heartbeat sounds and will use a regression approach to predict the 
location of the lub and dub sounds. Two neural network architectures will be tested: a convolutional neural network (CNN) and a fully connected neural network (FNN). 
The model will be evaluated using mean squared error for validation and accuracy calculation. The best model will be chosen by comparing the validation accuracy of
each model. The model will be implemented using Python and TensorFlow/Keras.
## Technologies Used
- Python
- TensorFlow/Keras
- Librosa
- Numpy
- Scikit-learn
    
## Neural Network Architecture
Two neural network architectures are used in this project:
- Convolutional Neural Network (CNN): A CNN is a type of deep learning model that is particularly 
well-suited for image and audio processing tasks. In this project, the CNN is used to extract 
features from the audio data that are relevant for predicting the location of the lub and dub sounds.

- Fully Connected Neural Network (FNN): An FNN, also known as a feedforward neural network, is a type 
of deep learning model that is composed of multiple layers of interconnected nodes. In this project, the 
FNN is used to learn the relationship between the extracted features and the location of the lub and dub sounds.

Both architectures were implemented using TensorFlow/Keras and trained on the dataset of segmented heartbeat sounds. 
The model is trained using mean squared error as the loss function
and accuracy is calculated using mean squared error. The best model architecture is chosen by comparing the test accuracy of each model.

## Model

1. During the training process, use a validation dataset to evaluate the model's performance at regular intervals (e.g. after every epoch). 
This can be done by setting aside a portion of the training data as validation data, and using this data to evaluate the model's accuracy during training.

2. Use the validation accuracy as a metric for monitoring the model's performance during training, and use this information to make adjustments to the model 
(e.g. by adjusting the learning rate, the number of layers, etc.) to improve its performance.

3. Once the training process is complete, use a test dataset to evaluate the model's performance and compare the test accuracy of different models. 
The model with the highest test accuracy is considered the best model.

4. After you've chosen the best model, you can use it on unseen data (data that was not used in training or validation) and compare the results to the test accuracy.

## Conclusion
The results of this project demonstrate the potential of machine learning for accurately predicting the location of lub and dub sounds in a heartbeat recording. 
This can have important implications for heart health diagnosis and monitoring. This can be used as a step towards further analysis of the heartbeat sounds and 
can be used in developing real-time monitoring and diagnostic systems.
