import pathlib

import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

"""THE AUTO MPG DATA SET"""
    
"""
if(__name__=="__main__"):
    main()

"""

def get_data():
    dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    print(dataset_path)

    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                    'Acceleration', 'Model Year', 'Origin']
    dataset = pd.read_csv(dataset_path, names=column_names,
                        na_values = "?", comment='\t',
                        sep=" ", skipinitialspace=True)

    print(dataset.tail())
    return dataset

def split_the_data(dataset):
    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    return train_dataset, test_dataset

def inspect_the_data(train_dataset):
    sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    #plt.show()

    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    print(train_stats)
    return train_stats

def split_labels_from_features(train_dataset,test_dataset):
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')
    return train_labels,test_labels

def normalize_the_data(train_stats,train_dataset,test_dataset):
    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']
    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)
    return normed_train_data,normed_test_data

def build_model(train_dataset):
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

def inspect_the_model(model,normed_train_data):
    print(model.summary())
    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)
    print(example_result)

def train_model(model,normed_train_data,train_labels):
    EPOCHS = 1000

    history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

    plotter.plot({'Basic': history}, metric = "mae")
    plt.ylim([0, 10])
    plt.ylabel('MAE [MPG]')
    plt.show()

    plotter.plot({'Basic': history}, metric = "mse")
    plt.ylim([0, 20])
    plt.ylabel('MSE [MPG^2]')
    plt.show()
    return EPOCHS

def train_early_stop_model(EPOCHS,model,normed_train_data,train_labels,normed_test_data,test_labels):
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    early_history = model.fit(normed_train_data, train_labels, 
                        epochs=EPOCHS, validation_split = 0.2, verbose=0, 
                        callbacks=[early_stop, tfdocs.modeling.EpochDots()])

    hist = pd.DataFrame(early_history.history)
    hist['epoch'] = early_history.epoch
    print(hist.tail())

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

    plotter.plot({'Early Stopping': early_history}, metric = "mae")
    plt.ylim([0, 10])
    plt.ylabel('MAE [MPG]')
    plt.show()

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
    return model

def make_predictions(model,normed_test_data,test_labels):
    test_predictions = model.predict(normed_test_data).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

    error = test_predictions - test_labels
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()

def main():
    dataset=get_data()

    """CLEAN THE DATA"""
    print(dataset.isna().sum())
    dataset = dataset.dropna()

    """
    this line cannot be performed in a separate method or else it will attempt 
    to map onto a copy of a splice of the dataset. Keras uses certain 
    underlying logic that's not reliable in copying values in that manner, 
    leading to the presence of NaN values in the dataset. To prevent this, this
    .map call is kept in the same scope as the dataset
    """
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
    print(dataset.tail())
    """END CLEAN THE DATA"""

    train_dataset,test_dataset=split_the_data(dataset)

    train_stats = inspect_the_data(train_dataset)
    train_labels, test_labels= split_labels_from_features(train_dataset,test_dataset)
    normed_train_data,normed_test_data=normalize_the_data(train_stats,train_dataset,test_dataset)
    model = build_model(train_dataset)
    inspect_the_model(model,normed_train_data)
    EPOCHS=train_model(model,normed_train_data,train_labels)

    model = build_model(train_dataset)
    model=train_early_stop_model(EPOCHS,model,normed_train_data,train_labels,normed_test_data,test_labels)
    make_predictions(model,normed_test_data,test_labels)

if(__name__=="__main__"):
    main()