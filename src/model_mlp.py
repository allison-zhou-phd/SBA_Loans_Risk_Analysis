import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score, roc_auc_score

from src.default_modeler import load_split_data

def get_weights(y_train):
    """
        calculate the weights used in fitting the mlp model
    Args:
        y_train: ndarray - 1D
    Returns:
        weight_for_0, weight_for_1 (float): two weights used in mlp model
    """
    counts = int(np.sum(y_train))
    total = len(y_train)
    print("Number of positive samples in training data: {} ({:.2f}% of total)".format(
          counts, 100*float(counts) / total))
    weight_for_0 = 1.0/(total-counts)
    weight_for_1 = 1.0/counts
    return weight_for_0, weight_for_1

def define_mlp_model(n_input):
    """ 
        define the multi-layer-perceptron neural network
    Args:
        n_input(int): number of features
    Returns:
        a defined mlp model, not fitted
    """
    model = Sequential()
    num_neurons = 256

    # hidden layer
    model.add(Dense(units=num_neurons,
                    input_dim=n_input,
                    kernel_initializer='he_uniform',
                    activation = 'relu'))
    model.add(Dense(units=num_neurons,
                    activation = 'relu')) 
    model.add(Dropout(0.3))
    model.add(Dense(units=num_neurons,
                    activation = 'relu')) 
    model.add(Dropout(0.3))
                   
    # output layer
    model.add(Dense(units=1,
                    activation = 'sigmoid'))
    model.summary()

    metrics = [FalseNegatives(name='fn'),
               FalsePositives(name='fp'),
               TrueNegatives(name='tn'),
               TruePositives(name='tp'),
               Precision(name='precision'),
               Recall(name='recall'),
    ]
    #sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
    adam = Adam(1e-2)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=metrics)
    return model

if __name__ == '__main__':
    (X_model, X_holdout, y_model, y_holdout), col_names = load_split_data(select=1)
    X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.1, random_state=42, stratify=y_model)
    
    ### Standardize features
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    X_holdout_std = scaler.transform(X_holdout)

    ### Building and tuning a neural network model
    n_input = X_std.shape[1]
    weight_for_0, weight_for_1 = get_weights(y_train)
    weights ={0:weight_for_0, 1:weight_for_1}
    
    ts = time()
    model = define_mlp_model(n_input)
    model.fit(X_std, y_train, epochs=30, batch_size=2048, verbose=2, 
              validation_data=(X_test_std, y_test), class_weight=weights)
    yhat = model.predict(X_holdout_std)
    score = roc_auc_score(y_holdout, yhat)
    te= time()
    print('ROC AUC: %.3f' % score)
    print("Time passed:", te-ts)

    save_model = 0
    if save_model:
        model.save("static/model_mlp.h5") 