# Language_Modeling

인공신경망과 딥러닝

Assignment #3

24510101 김상훈

shkim@ds.seoultech.ac.kr

# Abstract
This project implements two neural networks, **RNN and LSTM, for character-level language modeling using the Shakespeare dataset.** Both models will be trained and then used to generate samples of length 100. 

This report presents two main results:

(i) a comparison graph of the train and validation losses for both models over the training period 

(ii) an analysis of the impact of varying the softmax parameter \_T\_ on text generation.

# Report

## dataset
![dataset_args](./images/dataset_args.png)

### Requirements Met:
- **Character Indexing**: Constructs a dictionary mapping characters to indices and vice versa.
- **Sequence Preparation**: Splits the text data into sequences of length 30, with corresponding targets.

## model
![RNN](./images/RNN.png)
![LSTM](./images/LSTM.png)

### Description:

### Additional Techniques:


## main
![main_args1](./images/main_args1.png)
![main_args2](./images/main_args2.png)

## generate
![generate_args](./images/generate_args.png)

## Plots of loss values

## Report
