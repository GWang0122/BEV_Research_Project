import math
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow_datasets as tfds

from tensorflow import keras
#from keras import layers
import numpy as np
from vqvae import VQVAETrainer

image_height = 128
image_width = 256
latent_height = 32
latent_width = 64
num_epochs = 100
batch_size = 64

#parse images from tfrecord
def parse_images(example, img_height=image_height, img_width=image_width):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_png(example["image"], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [img_height, img_width])
    #image = tf.where(image != 1.0, 0.0, 1.0)


    return image


# Load TFRecord dataset
def create_image_dataset(file_path, batch_size, buffer_size):
    
    vehicles_dset = tf.data.TFRecordDataset(file_path)
    img_dset = vehicles_dset.map(parse_images, num_parallel_calls=tf.data.AUTOTUNE)
    img_dset = img_dset.batch(batch_size, drop_remainder=True)
    img_dset = img_dset.shuffle(buffer_size)
    img_dset = img_dset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return img_dset


def main():
    img_dset= create_image_dataset("train_data.tfrecord", batch_size=batch_size, buffer_size=640)
    scaled_images = np.concatenate(list(img_dset.as_numpy_iterator()), axis=0)
    data_variance = np.var(scaled_images)
    vqvae_trainer = VQVAETrainer(data_variance, latent_dim=16, num_embeddings=128)
    vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
    vqvae_trainer.fit(img_dset, epochs=num_epochs, batch_size=batch_size)
    vqvae_trainer.vqvae.save_weights("vqvae_weights.h5")
    

if __name__ == "__main__":
    main()




