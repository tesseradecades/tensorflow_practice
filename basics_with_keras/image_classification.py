import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#step 1
def load_data():
    fashion_mnist = keras.datasets.fashion_mnist
    return fashion_mnist.load_data()

#step 2
def eyeball_data(train_images,train_labels,test_images,test_labels):
    print(train_images.shape)
    print(len(train_labels))
    print(train_labels)
    print(test_images.shape)
    print(len(test_labels))
    #check the first image
    show_image(train_images[0])

def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()

# step 3
def normalize_data(train_images,test_images):
    train_images = train_images / 255.0

    test_images = test_images / 255.0

# step 4
def eyeball_normalization(train_images,class_names,train_labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

# step 5
def build_the_model():
    #set-up the layers
    model= keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)# 10 == the number of output categories == tthe number of class names
    ])

    #compile the model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

#step 6
def train_the_model(model,train_images,train_labels,test_images,test_labels,class_names):
    model.fit(train_images, train_labels, epochs=10)

    #evaluate accuracy
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    print(predictions[0])
    print(np.argmax(predictions[0]))
    print(test_labels[0])

    #verify predictions

    i = 0
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], test_labels, test_images,class_names)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  test_labels)
    plt.show()

    i = 12
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], test_labels, test_images,class_names)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  test_labels)
    plt.show()

    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], test_labels, test_images,class_names)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()

    return probability_model

def plot_image(i, predictions_array, true_label, img,class_names):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#step 7
def use_the_trained_model(test_images,probability_model,test_labels,class_names):
    # Grab an image from the test dataset.
    img = test_images[1]
    print(img.shape)

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img,0))

    print(img.shape)

    predictions_single = probability_model.predict(img)

    print(predictions_single)

    plot_value_array(1, predictions_single[0], test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    print(np.argmax(predictions_single[0]))
def main():
    print(tf.__version__)

    (train_images, train_labels), (test_images, test_labels) = load_data()
    print(train_images)

    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 
        'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    eyeball_data(train_images,train_labels,test_images,test_labels)

    normalize_data(train_images,test_images)
    
    eyeball_normalization(train_images,class_names,train_labels)

    model = build_the_model()

    probability_model=train_the_model(model,train_images,train_labels,test_images,test_labels,class_names)

    use_the_trained_model(test_images,probability_model,test_labels,class_names)

if(__name__=="__main__"):
    main()