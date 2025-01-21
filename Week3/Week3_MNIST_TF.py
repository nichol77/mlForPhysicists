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

# # Introduction to TensorFlow and Keras
# ## The MNIST dataset
# In this notebook we will introduce you to TensorFlow and Keras and we will demonstrate classification of one of the 'classic' datasets the MNIST handwritten digits.
#

# +
import matplotlib.pyplot as plt
import numpy as np
import math

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

import matplotlib.style #Some style nonsense
import matplotlib as mpl #Some more style nonsense


#Set default figure size
#mpl.rcParams['figure.figsize'] = [12.0, 8.0] #Inches... of course it is inches
mpl.rcParams["legend.frameon"] = False
mpl.rcParams['figure.dpi']=200 # dots per inch

#Useful for debugging problems
print(tf.__version__)
# -

# ## Load the MNIST dataset
#
# The Keras library comes with a number of builtin datasets. One of which is the MNIST library of handwritten digits

mnist = keras.datasets.mnist   #The original handwritten digit MNIST
#mnist = keras.datasets.fashion_mnist   #A tricky version that uses images of fashion items
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# ### Play with the data
# The training library consists of 60,000 images and each image is 28x28 pixels. There are 60,000 labels to go with the images. Each label is one of the digits (0:9).

print("Shape of training images:",train_images.shape)
print("Length of training set labels:",len(train_labels))
print("First label:",train_labels[0])
print("Shape of testing images:",test_images.shape)
print("Length of testing set labels:",len(test_labels))

# ### Plot an image
#
# Each image pixel has a value ranging from 0-255. When we come to do the processing we will rescale these inputs to the range 0-1.

plt.imshow(train_images[0])
plt.colorbar()

train_images=train_images/255.0
test_images=test_images/255.0


def display_image_array(whichImg):    
    numcols=int(math.sqrt(len(whichImg)))
    if numcols*numcols < len(whichImg):
        numcols=numcols+1
    BigImage=np.zeros([28*numcols,28*numcols])
    for j in range(len(whichImg)):
        x=(j%numcols)*28
        y=int(j/numcols)*28
        BigImage[x:x+28,y:y+28]=train_images[whichImg[j]]
    plt.imshow(BigImage,interpolation='nearest', cmap='binary')
    plt.axis('off')
    plt.show()


# +
whichImg=range(64)
display_image_array(whichImg)

#If you decide to use the fashion MNIST data set at some stage you will want to know the
# mapping from 0-9 to item of fashion
#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# -

# ## Building our neural network
#
# Now we will use Keras to build our neural network. 
#
# ### Set up the layers
#
# A neural network is made up of layers and in the simple case these layers are applied sequentially.

model = keras.Sequential([
    keras.layers.Input(shape=(28,28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10)
])

# This model has three layers:
# 1. The first layer simply 'flattens' our 28x28 pixel image into a 1-dimensional array of length 784 (=28x28). (Despite all the fancy features most neural networks are at heart one-dimensional in temrs of input*** of course higher dimensionality can always be squesihed down to a larger single dimension). This layer has no paramters.
# 2. The second layer is fully connected (or *dense*) layer with 128 nodes. This layer has parameters.
# 3. The third layer is the output layer and has 10 nodes which correspond to the digits 0:9. This layer has parameters.

# ## From layers to a compiled model
#
# Before we can train the model we need to determine the:
# - *Loss function* - This is used to tell us how well our model is doing in
# - *Optimiser* - This is how the model gets updated based on the data and the loss function
# - *Metrics* - Used to monitor the training and testing steps. The following example uses *accuracy*, the fraction of the images that are correctly classified.

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
              metrics=['accuracy'])

# ## Train the model
# To train the model we need to show the training data to the model. In our case the training data is the `train_images` and `train_labels` arrays. The model will hopefully *learn* how to associate the images and the labels.
#
# ### Fitting the model to the data
# The training process is also called fitting the model to the data (for obvious reasons). The Keras syntax for fitting the model uses `model.fit`

history=model.fit(train_images, train_labels,batch_size=100, epochs=30)

# In the above output you can see that the model improves the loss (this number gets smaller) and accuracy (this number gets bigger) with each epoch. But doing well on the training data set is a bit like marking your own homework. A fairer test is to evaluate the performance on the testing data set (which the model has not seen).

# ### Testing the performance

# +
print(np.shape(test_images))
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#print('\nTest accuracy:', test_acc)
# -

# So when I ran this I found a accuracy on the training dataset of 98.11% but the accuracy on the testing datset was 97.10%. Perfroming better on the training than testing datset is an indication of overfitting (which will be dicussed in more detail later in the course).

# ## Predictions with probabilities
# With our trained model we can now start trying to interogate the output probabilistically. We will make a new model with a *softmax* later to convert the linear outputs, [logits](https://developers.google.com/machine-learning/glossary#logits), to probabilities. N.B, We could have obtained the same by adding *softmax* activiation to the output layer in the original model definition. The *softmax* function is effectively taking an exponentiated unit function. So if the $i$th element of the *softmax* function $\sigma(x)$ is $$ \sigma(x)_i = \frac{e^{x_i}}{\sum_i e^{x_i}} $$
#

prob_model=tf.keras.Sequential([model,tf.keras.layers.Softmax()])


#This function tests a range of the test images and returns an array of the predicted values
#and the associate probabilities for those predictions
def test_range(start,stop,dontprint=False):
    global test_images, test_labels
    global prob_model
    #Get an array of the predicted probabilities for each image in the range start-stop (not inclusive)
    predictions_probs=prob_model.predict(test_images[start:stop,:]) 
    #Find the biggest probability for each image (the index of the array matches the numbers for the MNIST)
    #axis =1 tells np.argmax which axis to loop over
    predictions=np.argmax(predictions_probs,axis=1)
    #Optional printing
    if dontprint==False:
        print("Predictions: ", predictions)
    true_labels=test_labels[start:stop]
    if dontprint==False:
        print("True labels: ", true_labels)
    if dontprint==False:
        print("Prediction Probabilities: ", predictions_probs)
    #
    return (predictions,predictions_probs)


#Demonstrate the test_range function
_ = test_range(0,2)


# The output of the test_range function is the list of predicted and true labels for the range. For each prediction the associated probability array is also printed.

#This function shows the first maxnum mistakes in the testing dataset
def display_mistakes(maxnum):
    global test_images, test_labels
    (predictions, predictions_probs)=test_range(0,len(test_labels),dontprint=True)
    diff=test_labels-predictions
    count=0
    for j in range(len(test_labels)):
        if diff[j]!=0:
            if count<maxnum:
                count=count+1
                plt.imshow(test_images[j],interpolation='nearest', cmap='binary')
                plt.show()
                print("True ", test_labels[j], " - Predicted ", predictions[j], " with prob. ", predictions_probs[j,predictions[j]])
        


display_mistakes(10)

# ## Interrograting the training
# `keras.fit` returns a hitsory object that contains the evolution of the loss an accuracry metrics as a function of epoch. This allows us to see how quickly the fitting process is converging.

# fig,ax=plt.subplots()
# ax.plot(history.history['loss'], linewidth=3)
# ax.set_xlabel("Epoch")
# ax.set_ylabel("Loss")

fig,ax=plt.subplots()
ax.plot(history.history['accuracy'], linewidth=3)
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")

# ## Training, Validation and Testing
# Now we are going to break up the training data into a training and validation sub-sample to provide epoch-by-epoch performance monitoring

mnist = keras.datasets.mnist   #The original handwritten digit MNIST
#mnist = keras.datasets.fashion_mnist   #A tricky version that uses images of fashion items
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images=train_images/255.0  #See how big an effect the normalisation has by commenting out
test_images=test_images/255.0
(valid_images,valid_labels)=(train_images[50000:],train_labels[50000:])
(train_images,train_labels)=(train_images[:50000],train_labels[:50000])
print(np.shape(valid_images))
print(np.shape(train_images))
print(np.shape(train_labels))
print(np.shape(test_images))

# +
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(32,activation='relu'),
#    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
              metrics=['accuracy'])
# -

history=model.fit(train_images, train_labels,batch_size=100, epochs=200, validation_data=(valid_images, valid_labels))

# +
fig,ax=plt.subplots()
ax.plot(history.history['accuracy'], label='Train Acc.')
ax.plot(history.history['val_accuracy'], label = 'Validate Acc.')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
#ax.set_ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# -

# # Further work
# Now you can try and do different things with the keras network. Try and see if you can find a model that performs better on the training sample. Some possible optimisations include:
# - Switching from the standard MNIST dataset to the fashion MNIST dataset to see if it is harder to try and indentify shoes and t-shirts rather than numbers
# - Switching to a different optimisation routine (or changing the learning rate)
# - Changing the network architechture (number of layers or size of each layer)
# - Splitting the dataset into training and validation datasets to test the training is we progress
#


