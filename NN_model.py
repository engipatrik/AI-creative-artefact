# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 09:17:50 2023

@author: Patrik.Engi
"""

##
## This file contains the code to create and train the NN model by itself
##


# Importing the necessary libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from helper_functions import *

# Reading the preparing the data ahead of modelling
data=pd.read_csv('processed.csv')

# Due to the high number of gapfilled chemicals cholesterol is removed as an attribute
data=data.drop(columns=["Cholesterol"])

# Isolating the target classification 
y=data.pop("HeartDisease")

# Converting to easier to deal with data types
y=np.array(y)
x=np.array(data)

# Standardized data is required for better performance
X=normalise(x)

# Using the sklearn built in functionality to ensure that data is shuffled 
# as the order of data can have an impact on the model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=3, shuffle=True)

# Setting the seed so that results are reproduceable
tf.random.set_seed(42)

# Initialising a sequential model, with a the chosen architechture
model_10 = tf.keras.Sequential([
  tf.keras.layers.Dense(32,input_dim=14,kernel_initializer="HeNormal", activation="selu"),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dense(32, activation="relu"),
  tf.keras.layers.Dense(32, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model with the ideal learning rate
model_10.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(lr=0.001), 
                metrics=["AUC"])

# Fitting the model, note that an additional validation split of 20% is used
history = model_10.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Using the test dataset, the loss and accuracy scores are reported
loss, accuracy = model_10.evaluate(X_test, y_test)
print(f"Model loss on the test set: {loss}")
print(f"Model accuracy on the test set: {100*accuracy:.2f}%")


# Making predictions for all the test data 
test_preds = model_10.predict(X_test)

# Converting them from values in range [0,1] to binary classes 
test_preds = [round(float(i), 0) for i in test_preds]

# Calculating and displaying evaluation metrics
test_acc = accuracy_score(y_test, test_preds)
test_rec = recall_score(y_test, test_preds)
test_pre = precision_score(y_test, test_preds)
auc = roc_auc_score(y_test, test_preds)
print('Test Set Metrics')
print('Ensemble Model AUC score', auc)
print('Ensemble Model Accuracy:', test_acc)
print('Ensemble Model Recall:', test_rec)
print('Ensemble Model Precision:', test_pre)
print(confusion_matrix(y_test, test_preds))

# Displaying the architecture
model.summary()

# Plotting loss and auc plots
plot_loss_curves(history.history['loss'], history.history['val_loss'], 50, "loss")
plot_loss_curves(history.history['auc'], history.history['val_auc'], 50, "auc")

# Plotting the AUC & ROC curve
plot_auc(y_test, test_preds )

# Plotting heatmap 
confusion_heatmap(y_test, test_preds)
    

    
    