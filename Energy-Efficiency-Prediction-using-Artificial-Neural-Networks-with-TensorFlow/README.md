# -Energy-Efficiency-Prediction-using-Artificial-Neural-Networks-with-TensorFlow

This repository contains code for building and training an Artificial Neural Network (ANN) using TensorFlow and Keras. The ANN is designed to predict energy efficiency based on a dataset. The process involves data preprocessing, model building, training, and evaluation.

Dataset
The dataset used is 'Folds5x2_pp.xlsx', which contains various features related to energy efficiency. Download and place this dataset in the same directory as the script.

Code Overview
1. Importing the Libraries
The script starts by importing necessary libraries such as NumPy, Pandas, TensorFlow, and scikit-learn.

2. Data Preprocessing
Importing the Dataset
The dataset is loaded using Pandas. Features and the target variable are extracted into separate arrays.

Splitting the Dataset
The data is split into training and test sets using train_test_split from scikit-learn to enable model training and evaluation.

3. Building the ANN
Initializing the ANN
A sequential model is initialized using TensorFlow's Keras API.

Adding Layers
Input and First Hidden Layer: A dense layer with 6 units and ReLU activation function is added.
Second Hidden Layer: Another dense layer with 6 units and ReLU activation function is added.
Output Layer: A dense layer with 1 unit (since it is a regression problem) is added.

4. Training the ANN
Compiling the Model
The model is compiled with the Adam optimizer and mean squared error as the loss function.

Training
The model is trained on the training set for 100 epochs with a batch size of 32.

5. Making Predictions
After training, the model's performance is evaluated by making predictions on the test set. The predictions are then printed alongside the actual values for comparison.

This code provides a straightforward implementation of an ANN for regression tasks, demonstrating key steps from data preprocessing to model evaluation.
