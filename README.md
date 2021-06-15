# Introduction
In this notebook, I will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline! My completed pipeline will accept raw audio as input and return a predicted transcription of the spoken language. 
The full pipeline is summarized in the figure below.

It has main three steps:
STEP 1 is a pre-processing step that converts raw audio to one of two feature representations that are commonly used for ASR.
STEP 2 is an acoustic model which accepts audio features as input and returns a probability distribution over all potential transcriptions. 
       After learning about the basic types of neural networks that are often used for acoustic modeling, I will engage in my own investigations, to design my own acoustic model!
STEP 3 in the pipeline takes the output from the acoustic model and returns a predicted transcription.

# Models
I have bulilt many models to see their performance on the task of automatic speech recognition as follows:
I used both spectrogram and Mel-Frequency Cepstral Coefficients (MFCCs) features in training my models,

- Simple RNN using GRU which conists of one hiddien layer, and one output layer which is activated using softmax. 
  The performance in this Simple RNN was so bad on both spectrogram and Mel-Frequency Cepstral Coefficients (MFCCs) features.

- Rnn model which includes:
  RNN using GRU which conists of one hiddien layer, BatchNormalization, TimeDistributed, output layer.
  
  What is Batch normalization (BN)? How it works?
  1st: how the BN works:
  Nomalization basically is used to convert all values into values between 0 and 1 to speed up learning.
  Batch normalization is done not with inputs but with hidden units. 
  it convert the value comes from the layer before normalization into value between 0 and 1. 
  The BN layer takes the input from previous layer before it and subtract this previous layer output from mean and divide it by standard deviation.
  
  2nd Why is BN used?
  To increase the learning rate and also to increase the robusness of the network. to know more please see this video:
  https://www.youtube.com/watch?v=EvAVCxZJN2U
  
  TimeDistributed: 
  A time distributed vector just applies the same function to every time step. Its not a recurrent function like an LSTM; the TD layer looks at each time step on its own, with out considering any other time steps.
  This can be used when you want to organize some data like a sequence, but you don't want to consider things in any specific order. If you apply TD(Dense) layers and then add along the time axis, 
  that output will not change if you shuffle the order of the input time steps. To know more about this:
  https://www.reddit.com/r/learnmachinelearning/comments/g75015/could_anyone_please_explain_timedistributed_layer/
  
- CNN, RNN model which includes: convolution, batch normalization, sinmple RNN layer, batch normalization, and time distribution.

- Deep RNN which consists of more RNN layers.

- Bidirectional RNN which includes one bidirectional layer with time distributed layer.

- Final Model: which includes CNN, BN, 2 RNN, 1 time distribution layer.  

# Setup

This project requires GPU acceleration to run efficiently. 
Support is available to use either of the following two methods for accessing GPU-enabled cloud computing resources.

- Create a new enviroment use: create conda -n mt_env
- install all required packages inside env.txt file, pip install -r env.txt
- run the jupyter note book cells.

## Install
- Python 3
- NumPy
- TensorFlow 1.x
- Keras 2.x

- Notice, I used the Udacity workspace after enabling GPU to execute the project. I tried to execute it in my lab, I couldn't because of lacking GPU.
