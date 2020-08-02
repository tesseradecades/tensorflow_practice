import matplotlib.pyplot as plt
import os
import re
import shutil
import string
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

"""SENTIMENT ANALYSIS"""

#step 1
def download_data_set():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                        untar=True, cache_dir='.',
                                        cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    print(os.listdir(dataset_dir))

    train_dir = os.path.join(dataset_dir, 'train')
    print(os.listdir(train_dir))

    sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
    with open(sample_file) as f:
        print(f.read())
    return train_dir

#step 2
def load_the_data_set(train_dir):
    #remove unneeded folders
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)
    
    # create validation set
    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/train', 
        batch_size=batch_size, 
        validation_split=0.2, 
        subset='training', 
        seed=seed)
    
    # explore the data
    for text_batch, label_batch in raw_train_ds.take(1):
        for i in range(3):
            print("Review", text_batch.numpy()[i])
            print("Label", label_batch.numpy()[i])
    print("Label 0 corresponds to", raw_train_ds.class_names[0])
    print("Label 1 corresponds to", raw_train_ds.class_names[1])

    # validation data set
    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/train', 
        batch_size=batch_size, 
        validation_split=0.2, 
        subset='validation', 
        seed=seed
    )

    # test data set
    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/test', 
        batch_size=batch_size
    )
    return raw_train_ds,raw_val_ds,raw_test_ds

#step 3
def prepare_data_set_for_training(raw_train_ds,raw_val_ds,raw_test_ds):
    max_features = 10000
    sequence_length = 250

    # create vectorization layer
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    # retrieve a batch (of 32 reviews and labels) from the dataset
    text_batch, label_batch = next(iter(raw_train_ds))
    first_review, first_label = text_batch[0], label_batch[0]
    print("Review", first_review)
    print("Label", raw_train_ds.class_names[first_label])
    print("Vectorized review", vectorize_text(first_review, first_label))

    #explore the vocabulary
    print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
    print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
    print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)
    return max_features,train_ds,val_ds,test_ds,vectorize_layer

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(
      stripped_html,
      '[%s]' % re.escape(string.punctuation),
      ''
    )

#step 4
def configure_data_set_for_performance(train_ds,val_ds,test_ds):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# step 5
def create_the_model(max_features):
    embedding_dim = 16
    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    model.summary()
    return model

# step 6
def loss_function_and_optimizer(model):
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# step 7
def train_the_model(model,train_ds,val_ds):
    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    return history

# step 8
def evaluate_the_model(model,test_ds):
    loss, accuracy = model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

# step 9
def plot_accuracy_and_loss_over_time(history):
    history_dict = history.history
    history_dict.keys()

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()

"""EXPORT THE MODEL"""
#step 10
def export_the_model(vectorize_layer,model,raw_test_ds):
    export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )

    # Test it with `raw_test_ds`, which yields raw strings
    loss, accuracy = export_model.evaluate(raw_test_ds)
    print(accuracy)

def main():
    print(tf.__version__)

    train_dir=download_data_set()
    #train_dir=os.path.abspath("aclImdb\\train")
    print(train_dir)

    raw_train_ds,raw_val_ds,raw_test_ds= load_the_data_set(train_dir)

    max_features,train_ds,val_ds,test_ds,vectorize_layer=prepare_data_set_for_training(raw_train_ds,raw_val_ds,raw_test_ds)

    configure_data_set_for_performance(train_ds,val_ds,test_ds)

    model=create_the_model(max_features)

    loss_function_and_optimizer(model)

    history=train_the_model(model,train_ds,val_ds)

    evaluate_the_model(model,test_ds)

    plot_accuracy_and_loss_over_time(history)

    export_the_model(vectorize_layer,model,raw_test_ds)

if(__name__=="__main__"):
    main()