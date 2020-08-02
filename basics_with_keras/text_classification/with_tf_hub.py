import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

# step 1
def download_data_set():
    # Split the training set into 60% and 40%, so we'll end up with 15,000 examples
    # for training, 10,000 examples for validation and 25,000 examples for testing.
    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews", 
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True)
    return train_data, validation_data, test_data

# step 2
def explore_the_data(train_data):
    train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
    print(train_examples_batch)
    print(train_labels_batch)
    return train_examples_batch

# step 3
def build_the_model(train_examples_batch):
    embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                            dtype=tf.string, trainable=True)
    print(hub_layer(train_examples_batch[:3]))

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.summary()

    loss_function_and_optimizer(model)
    return model

def loss_function_and_optimizer(model):
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

# step 4
def train_the_model(model,train_data,validation_data):
    history = model.fit(
        train_data.shuffle(10000).batch(512),
        epochs=20,
        validation_data=validation_data.batch(512),
        verbose=1
    )

# step 5
def evaluate_the_model(model,test_data):
    results = model.evaluate(test_data.batch(512), verbose=2)

    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))

def main():
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

    train_data, validation_data, test_data = download_data_set()

    train_examples_batch=explore_the_data(train_data)

    model=build_the_model(train_examples_batch)

    train_the_model(model,train_data,validation_data)

    evaluate_the_model(model,test_data)

if(__name__=="__main__"):
    main()