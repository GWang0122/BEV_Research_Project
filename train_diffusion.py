#!/usr/bin/env python

# In[6]:


import math
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import diffusion
from diffusion import DiffusionModel
import unet
from vqvae import VQVAETrainer
from train_vqvae import create_image_dataset
import os

# In[7]:


# data
dataset_repetitions = 5
num_epochs = 4000  # train for at least 50 epochs for good results
image_height = 128
image_width = 256
latent_height = 32
latent_width = 64

diffusion_steps = 5
plot_diffusion_steps = 20

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 32
emb_size=32
#num_classes = 12

widths = [32, 64, 96]
block_depth = 2
attention_levels = [0, 1, 0]
latent_dim = 16

# optimization
batch_size = 64
ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4


# In[8]:


img_dset = create_image_dataset("train_data.tfrecord", batch_size=64, buffer_size=640)
#val_dset = create_image_dataset("valid_data.tfrecord", batch_size=64, buffer_size=640)
scaled_images = np.concatenate(list(img_dset.as_numpy_iterator()), axis=0)
data_variance = np.var(scaled_images)


# In[9]:


class PlotImagesCallback(keras.callbacks.Callback):
    def __init__(self, model, img_dset, num_rows, num_cols):
        super().__init__()
        self.model = model
        self.img_dset = img_dset
        self.num_rows = num_rows
        self.num_cols = num_cols

    def on_epoch_end(self, epoch, logs=None):
        # Get a single batch from the dataset
        batch_images = next(iter(self.img_dset))
        #print(batch_images)

        #print(random_images.shape)
        # Plot images before forward diffusion and after reverse diffusion
        self.model.plot_images(
            batch_images=batch_images,
            epoch=epoch,
            num_rows=self.num_rows,
            num_cols=self.num_cols
        )


# In[10]:


vqvae = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128)
vqvae.vqvae.load_weights("vqvae_weights.h5")



# create and compile the model
model = DiffusionModel(widths,
                       block_depth,
                       attention_levels,
                       vqvae)

plot_images_callback = PlotImagesCallback(model, img_dset, num_rows=2, num_cols=2)

# below tensorflow 2.9:
# pip install tensorflow_addons

import tensorflow_addons as tfa
model.compile(
    optimizer=tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    loss=keras.losses.mean_absolute_error,
)
'''

model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    loss=keras.losses.mean_absolute_error,
)
'''
#plot model
#keras.utils.plot_model(model.network, "model.png")

# pixelwise mean absolute error is used as loss

# save the best model based on the validation KID metric
checkpoint_path = "checkpoints/diffusion_chkpt"

#check if model checkpoint exists. If model checkpoint exists, then load the checkpoint. 
start_epoch = 0
if(os.path.exists(checkpoint_path)):
    print("successfully found checkpoint path, training from checkpoint")
    model.load_weights(checkpoint_path)
    start_epoch += 500
    

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="n_loss",
    mode="min",
    save_best_only=True,
)

# calculate mean and variance of training dataset for normalization
#model.normalizer.adapt(img_dset)


# run training and plot generated images periodically
model.fit(
    img_dset,
    epochs=num_epochs,
    initial_epoch=start_epoch,
    #steps_per_epoch=100,
    #validation_data=val_dataset,
    callbacks=[
        plot_images_callback,
        checkpoint_callback,
    ],
)







