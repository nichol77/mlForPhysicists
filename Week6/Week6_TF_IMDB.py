# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Natural Language Processing
#
# In this notebook we introudce some of the ideas of natural language processing. To speed things up ww will be using a couple of optional Tensorflow libraries. Tensorflow Hub and Tensoflow Datasets and also a library called tf_keras which is basically the same as the keras already in tensorflow.
#
# The content of this notebook is extremely heavily inspired by one of the [Tensorflow tutorials](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub)
#
# # Pre-trained models
#
# This notebook is the first time in the course that we will be using a pre-trained model (in this case for a text embedding layer). Using these pre-trained models can dranatically speed up certain tasks.
#
# ## Installing the libraries
# In the first cell we use pip to install these libraries.

# !pip install tensorflow_hub
# !pip install tensorflow_datasets
# !pip install tf_keras

# ## Import libraries, check versions and GPU availability
# Next up we import the libraries and check some of the Tensorflow settings and see if you are running on a GPUs or not.

# +
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tf_keras as keras


print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")



# -

# ## Load Internet Movie Database Review Dataset
# Now we are going to load a text dataset from the [Internet Movie Database](https://www.imdb.com). This dataset consists of reviews of various lengths of text that have been labelled as either positive or negative.

# Split the training set into 60% and 40% to end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

# ## Printing some example reviews
# The Tensorflow dataset is an interesting object, in the code below we create a batch of 10 reviews (and labels) and then turn it into a Tensorflow Iterator with `iter` and actually get the values with `next`. At first it might seem a little clunky using tensorflow datasets but they are useful resource to try and understand.

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)

#Also print the labels
print(train_labels_batch)

# ## Text Embedding
# Obvioulsy as we've stated repeatedly in the course neural networks take input vectors (of various sizes) and then apply some combination of matrix multiplications, additions, non-linear activation functions and other mathematical operations. Normally English sentences are not immediately well-suited to these operations so the first stage in any natural language processing is to convert the input into some kind of numerical vector via a process of embedding. Fortunately lots of people have done this before us so in true [Blue Peter](https://www.bbc.co.uk/cbbc/watch/bp-heres-one-i-made-earlier-video-challenge) style we can use one that is already been pre-trained.
#
# In the code below we use Tensorflow Hub to get a pre-trained text embedding layer. The embedding we use is the catchily named [google/nnlm-en-dim50/2](https://tfhub.dev/google/nnlm-en-dim50/2) which as it's name suggest is Neural Network Langauge Model in English which encodes sequences into vectors of dimension 50. We create a layer called hub_layer which is trainable but starts off from the already pre-trained weights.
#
# #### From TensorFlow's Tutorial
# There are many other pre-trained text embeddings from TFHub that can be used in this tutorial:
#
# * [google/nnlm-en-dim128/2](https://tfhub.dev/google/nnlm-en-dim128/2) - trained with the same NNLM architecture on the same data as [google/nnlm-en-dim50/2](https://tfhub.dev/google/nnlm-en-dim50/2), but with a larger embedding dimension. Larger dimensional embeddings can improve on your task but it may take longer to train your model.
# * [google/nnlm-en-dim128-with-normalization/2](https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2) - the same as [google/nnlm-en-dim128/2](https://tfhub.dev/google/nnlm-en-dim128/2), but with additional text normalization such as removing punctuation. This can help if the text in your task contains additional characters or punctuation.
# * [google/universal-sentence-encoder/4](https://tfhub.dev/google/universal-sentence-encoder/4) - a much larger model yielding 512 dimensional embeddings trained with a deep averaging network (DAN) encoder.
#
# And many more! Find more [text embedding models](https://tfhub.dev/s?module-type=text-embedding) on TFHub.
#
#

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

# ## Building a full model
# To build the full model we need to take our embedding layer and then add a simple fully connected network that ends with a single neuron in the output layer (which we will use to compare to the labels). 

# +
model = keras.Sequential()
model.add(hub_layer)
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1))

model.summary()
# -

# ## Model comments
#
# So our full model contains 48,191,433 parameters. But the vast majority of these are in the form of the embedding layer and those parameters are already pre-trained. The fully connected classification part of the network is just 816 + 17 parameters.
#
# # Model Compilation
#
# We are going to use the `adam` optimizer and since we are doing a binary classification taks we will use BinaryCrossentropy as our loss function.
#
# # Model training
#
# Then we will train our model for 10 epochs. To speed things up we will only train on a subset of the data.

# +
model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validation_data.batch(512),
                    verbose=1)
# -

# ## Model performance
#
# The last step is to use our test dataset (which we haven't touched yet) to evaluate the performance of the model.
#
# When I first ran this the accuracy was 85.3% which is not bad considering all we have done is add a single hidden layer of 16 neurons after our pre-trained embedding layer.

# +
results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))
# -

# # Suggested tasks
# 1. Can you improve on the model performance?
# 2. How does the speed of model training change if you try some of the other pre-trained text embeddings layers?


