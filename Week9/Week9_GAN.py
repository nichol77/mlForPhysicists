# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/nichol77/mlForPhysicists/blob/master/Week9/Week9_GAN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="I0mGs2zA6kBm"
# # Practical Machine Learning for Physicsists
#
# ## Generative Adversarial Networks
#
# In this notebook we will introduce you to generative adversarial networks. In the time honoured tradition we will do this through the MNIST dataset (our favourite toy dataset).
#
# This notebook is heavily inspired by the [TensorFlow tutorial on Deep Convolutional Generative Adversarial Networks.](https://www.tensorflow.org/tutorials/generative/dcgan). The code is basically the same as in the turorial although the commentary is slightly different. 

# + colab={"base_uri": "https://localhost:8080/"} id="lBqSSpEW6kBm" outputId="020f5760-014a-43e0-b2f0-e14cb18a2d82"
import matplotlib.pyplot as plt
import numpy as np
import math

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

import matplotlib.style #Some style nonsense
import matplotlib as mpl #Some more style nonsense

import glob
import imageio
import os
import PIL
import time

from IPython import display

#Set default figure size
#mpl.rcParams['figure.figsize'] = [12.0, 8.0] #Inches... of course it is inches
mpl.rcParams["legend.frameon"] = False
mpl.rcParams['figure.dpi']=200 # dots per inch

#Useful for debugging problems
print(tf.__version__)

# + [markdown] id="6sidlmdh6kBn"
# ## Step 1: Load and prepare the data
# We first load and normalise the MINST handwritten digits dataset. Then we shuffle and batch up the dataset for easier digestion by TensorFlow.
#
# Note that we are only using the training portion of the dataset, the testing portion isn't needed.

# + id="6LJA0uBQ6kBn"
# Load the data
mnist = keras.datasets.mnist   #The original handwritten digit MNIST
#mnist = keras.datasets.fashion_mnist   #A tricky version that uses images of fashion items
(train_images, train_labels), (_, _) = mnist.load_data()

# Reshape the data
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]


# + id="DqtyJYf86kBn"
# Example of tf dataset
BUFFER_SIZE = 60000
BATCH_SIZE = 256
NOISE_LENGTH = 100
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# + [markdown] id="lGarHr8y6kBn"
# ## Step 2: Create our generator model
# The generator model will try and turn random numbers into handwritten digit images. It will do this through a series of deconvolution and normalisation layers.
# - [`tf.keras.layers.Conv2DTranspose`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose) a transposed convolution layer (i.e. deconvolution)
# - [`tf.keras.layers.BatchNormalization`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization) normalize and scale inputs and activations by subtracting the mean and dividing by the standard deviation
# - [`tf.keras.layers.LeakyReLU`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LeakyReLU) a leaky version of ReLU, leaky means that for $x<0$ there is a small (i.e. non-zero) gradient
#
#

# + id="0quu5cGP6kBn"
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(NOISE_LENGTH,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


# + colab={"base_uri": "https://localhost:8080/"} id="J3CuTu2Z6kBn" outputId="8a30dc7b-d38e-4768-c8f6-00c1562dcde5"
generator=make_generator_model()
generator.summary()

# + colab={"base_uri": "https://localhost:8080/", "height": 785} id="Mwn-p6fN6kBn" outputId="fe604fa4-d618-4e12-ce2b-ee6568e643a1"
input=tf.random.normal([1,NOISE_LENGTH])
generated_image = generator(input, training=False)
print(generated_image.shape)

fig,ax = plt.subplots()
ax.imshow(generated_image[0,:,:,0],cmap='binary')
ax.set_title("Generator before training")


# + [markdown] id="1hgkKXdz6kBo"
# ## Step 3: Make discriminator model
# The discriminator model is going to be the standard Convolutional neural network.

# + id="MJ4DJzA26kBo"
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


# + [markdown] id="zLQ-tY0E6kBo"
# The discriminator is used to classify images as real or fake. The model will be trained to output positive values for real images, and negative values for fake images.

# + colab={"base_uri": "https://localhost:8080/"} id="CLsHifr66kBo" outputId="d7043611-9cbd-48d9-9580-be845fba759b"
discriminator = make_discriminator_model()
discriminator.summary()
decision = discriminator(generated_image)
print ("The decision on noise images:",decision)

# + [markdown] id="hRnLpsvT6kBo"
# ## Step 4: Define the loss
# Now we define the loss, separately for both the discriminator and the generator.

# + id="zmqVe1ih6kBo"
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# + [markdown] id="r7s0AJak6kBo"
# ### Discriminator loss
# The discriminator loss is the measure of how well the discriminator can distinguish real images from fake images. So:
# - The loss for real images is computed by comparing the discriminator output on real images against an array of 1's
# - The loss for fake images is computed by comparing the discriminator output on fake images against an array of 0's

# + id="pgzzsPc-6kBo"
def discriminator_loss(real_output, fake_output):
    return cross_entropy(tf.ones_like(real_output),real_output) + cross_entropy(tf.zeros_like(fake_output),fake_output)


# + [markdown] id="LQw29o8K6kBo"
# ### Generator loss
# The generator loss is the measure of well the generator can fool the discriminator into thinking that fake images are real. So
# - The loss is computed by comparing the discriminator output on fake images against an array of 1's

# + id="BgEvkrxI6kBo"
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output)


# + [markdown] id="Nyl5PlCs6kBo"
# # Step 5: Define the optimisers
# We will just use Adam

# + id="lxP4BDgA6kBo"
gen_optimiser=tf.keras.optimizers.Adam(1e-4)
dis_optimiser=tf.keras.optimizers.Adam(1e-4)

# + [markdown] id="nCsStvEi6kBo"
# # Step 6: Use checkpoints to save progress during training
# As we scale to bigger models that take longer to train it makes sense to start thinking about how we can save the results of our training for later use. If we save during the training then this can also help if our training becomes interuppted.

# + id="cXeEToJT6kBo"
import os
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimiser,
                                 discriminator_optimizer=dis_optimiser,
                                 generator=generator,
                                 discriminator=discriminator)

# + [markdown] id="9lnGRrnZ6kBo"
# # Step 7: Define the training loop
#
# The training loop should be something like:
# 1. Loop over epochs
#     2. Loop over `BATCH_SIZE` batches of real images in training dataset
#         1. Get `BATCH_SIZE` * `NOISE_LENGTH` random numbers
#         2. Generate `BATCH_SIZE` fake images
#         3. Run the discriminator over the real images
#         4. Run the discrimnator over the fake images
#         5. Compute the discriminator loss
#         6. Compute the generator loss
#         7. Get the gradients
#         8. Update using the optimisers
#
# Below we will use two forms of TensorFlow magic optimisation:
# - [`@tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) - 'compiles' the code for faster operation
# - [`tf.GradientTape`](https://www.tensorflow.org/api_docs/python/tf/GradientTape) - Used to record operations for automatic differentiation  (look at the documentation link for further details)

# + id="xUxOjKWU6kBo"
EPOCHS = 50
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
test_seed = tf.random.normal([num_examples_to_generate, NOISE_LENGTH])

gen_loss_array=np.zeros(EPOCHS)
disc_loss_array=np.zeros(EPOCHS)
gen_loss_err=np.zeros(EPOCHS)
disc_loss_err=np.zeros(EPOCHS)


# + id="4NGGJLAk6kBo"
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_LENGTH])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimiser.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    dis_optimiser.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss,disc_loss


# + id="5gUUzPco6kBo"
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        
        N=len(dataset)
        gloss=np.zeros(N)
        dloss=np.zeros(N)
        i=0
        for image_batch in dataset:
            gl,dl=train_step(image_batch)
            gloss[i]=gl
            dloss[i]=dl
            i+=1

        gen_loss_array[epoch]=np.mean(gloss)
        disc_loss_array[epoch]=np.mean(dloss)
        gen_loss_err[epoch]=np.std(gloss)
        disc_loss_err[epoch]=np.std(dloss)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 test_seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                            epochs,
                            test_seed)


# + [markdown] id="JqKq8o3H6kBo"
# #### Generate and save images to produce an animated gif

# + id="fwSvs1oN6kBo"
def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# + [markdown] id="CBX88ZfC6kBo"
# # Step 8: Do the training
# Now that we have set up all the training loops we can try and train our networks. Training GANs is challenging. As the two networks battle against each other it is important that they learn at a similar rate, if one gets on top too early than neither will learn efficiently.
#
# This step is slow, on my (not so good) laptop it takes almost three minutes per epoch. Running on Google Colab without GPUs the it was over 10 minutes per epoch. But running on Google Colab with [GPUs enabled](https://colab.research.google.com/notebooks/gpu.ipynb) it took about 12 seconds per epoch. A speedup of almost a factor of 60! Of course your mileage may vary, or you may have access to local GPU resources.

# + colab={"base_uri": "https://localhost:8080/", "height": 661} id="_YM745dm6kBo" outputId="e80addbd-463f-4595-92fd-7b07fcc80fbd"
train(train_dataset, EPOCHS)

# + [markdown] id="rqrpDX_f6kBo"
# #### Restore the checkpoint

# + colab={"base_uri": "https://localhost:8080/"} id="xZmaz3Hc6kBo" outputId="38d34a5f-fee5-4843-98c2-f2e65e03a62d"
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# + [markdown] id="vPm1KY_h6kBo"
# # Step 9: Make an animated GIF
#

# + id="HdJpWVli6kBo"
anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

# + [markdown] id="ko82Iu_96kBo"
# # Step 10: Look at the gif
# There are ways to display the GIF from the python code, or you can just do it using Markdown like I have done here.
# ![Numbers](dcgan.gif "numbers")

# + id="LHi_7WM36kBo"
#Here is an example of the same thing but from the python cell using IPython
#from IPython.display import Image
#Image(filename="dcgan.gif")

# + [markdown] id="znx-dDq36kBo"
# ## Step 11: Look at the loss vs epoch training graph
# It is interesting to realise that our old standby for neural network training success, the plot of loss vs training time, is no longer a useful a source of information. Since at the start neither the discriminator nor the generator knows anything they are evenly matched. As the training develops and the quality of the generation improves, so does the quality of the discrimination. How do you determine success in such a scenario?

# + colab={"base_uri": "https://localhost:8080/", "height": 764} id="F_Bkz9H66kBo" outputId="3e303c52-f230-4eeb-c9c0-7edc18ef3222"
fig,ax=plt.subplots()
epochs=np.arange(EPOCHS)
ax.errorbar(epochs,gen_loss_array,yerr=gen_loss_err,label="Generator Loss")
ax.errorbar(epochs,disc_loss_array,yerr=disc_loss_err,label="Discriminator Loss")
ax.set_xlabel("Training Epoch")
ax.set_ylabel("Loss")
ax.legend()
