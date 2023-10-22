import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from unet import get_network
import numpy as np

# data
dataset_repetitions = 5
num_epochs = 500  # train for at least 50 epochs for good results
image_height = 128
image_width = 256
latent_height = 32
latent_width = 64

diffusion_steps = 1000
plot_diffusion_steps = 1000

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

class DiffusionModel(keras.Model):
    def __init__(self, widths, block_depth, attention_levels, vqvae):
        super().__init__()

        #self.normalizer = layers.Normalization()
        self.network = get_network(latent_height,
                                   latent_width,
                                   widths,
                                   block_depth,
                                   attention_levels)
        self.ema_network = keras.models.clone_model(self.network)
        self.vqvae = vqvae
        self.encoder = self.vqvae.get_encoder()
        self.decoder = self.vqvae.get_decoder()
        self.quantizer = self.vqvae.quantize()




    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.latent_loss_tracker = keras.metrics.Mean(name="l_loss")


    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.latent_loss_tracker]

    #def denormalize(self, latents):
        # convert the pixel values back to 0-1 range
        #latents = self.normalizer.mean + latents * self.normalizer.variance**0.5
        #return tf.clip_by_value(latents, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_latents, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        #save input for plotting model
        pred_noises = network([noisy_latents, noise_rates**2], training=training)
        pred_latents = (noisy_latents - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_latents

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_latents = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_latents = initial_noise
        for step in range(diffusion_steps):
            noisy_latents = next_noisy_latents

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_latents, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_latents = self.denoise(
                noisy_latents, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_latents = (
                next_signal_rates * pred_latents + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_latents

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, latent_height, latent_width, latent_dim))
        generated_latents = self.reverse_diffusion(initial_noise, diffusion_steps)
        #generated_latents = self.denormalize(generated_latents)
        generated_images = self.decoder(generated_latents)
        return generated_images

    def train_step(self, images):
        latents = self.encoder(images)
        #latents = self.normalizer(latents, training=True)
        latents = self.quantizer(latents)
        # normalize images to have standard deviation of 1, like the noises

        noises = tf.random.normal(shape=(batch_size, latent_height, latent_width, latent_dim))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

       # print("signal_rates shape:", signal_rates.shape)
       # print("images shape:", images.shape)
       # print("noises shape:", noises.shape)
        # mix the images with noises accordingly
        noisy_latents = signal_rates * latents + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_latents = self.denoise(
                noisy_latents, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            latent_loss = self.loss(latents, pred_latents)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.latent_loss_tracker.update_state(latent_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        latents = self.encoder(images)
        #latents = self.normalizer(latents, training=False)
        latents = self.quantize(latents)
        # normalize images to have standard deviation of 1, like the noises

        noises = tf.random.normal(shape=(batch_size, latent_height, latent_width, latent_dim))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_latents = signal_rates * latents + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_latents = self.denoise(
            noisy_latents, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        latent_loss = self.loss(latents, pred_latents)

        self.latent_loss_tracker.update_state(latent_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        #pred_latents = self.denormalize(pred_latents)

        generated_images = self.generate(
            num_images=batch_size, diffusion_steps=diffuse_steps
        )
        #self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}
        
    def forward_backward_diffusion(self, images, num_diffusion_steps):
        ## Forward diffusion
        latents = self.encoder(images)
        latents = self.quantizer(latents)
        noises = tf.random.normal(shape=(batch_size, latent_height, latent_width, latent_dim))
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        #print(noise_rates.shape)
        #print(latents.shape)
        noisy_latents = signal_rates * latents + noise_rates * noises

        # Backward diffusion
        recon_latents = self.reverse_diffusion(noisy_latents, num_diffusion_steps)
        recon_images = self.decoder(recon_latents)
        return recon_images

    def plot_images(self, batch_images, epoch=None, logs=None, num_rows=2, num_cols=2):
        # Plot 2 original and 2 reconstructed images from the diffusion model

        # Reconstruct with the diffusion model
        generated_images = self.forward_backward_diffusion(
            images=batch_images,
            num_diffusion_steps=plot_diffusion_steps,
        )

        num_images = num_rows * num_cols
        num_samples = 2  # Two original and two generated images
        random_indices = np.random.choice(len(batch_images), num_samples, replace=False)
        random_images = tf.gather(batch_images, random_indices)
        random_generated_images = tf.gather(generated_images, random_indices)

        plt.figure(figsize=(num_cols * 3.0, num_rows * 3.0))
    
        for i in range(num_images):
            plt.subplot(num_rows, num_cols, i + 1)

            # Plot either original or generated image
            if i < num_images // 2:
                plt.imshow(random_images[i])
                plt.title("Original")
            else:
                plt.imshow(random_generated_images[i - num_images // 2])
                plt.title("Generated (Diffusion)")

            plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"diffusion_results/epoch_{epoch + 1:03d}.png")
        plt.close()


        
    def display_generate(self, epoch=None, logs=None, num_rows=2, num_cols=2):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=plot_diffusion_steps,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()
