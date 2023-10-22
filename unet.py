from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import math

def attention(qkv):

    q, k, v = qkv
    # should we scale this?
    s = tf.matmul(k, q, transpose_b=True)  # [bs, h*w, h*w]
    beta = tf.nn.softmax(s)  # attention map
    o = tf.matmul(beta, v)  # [bs, h*w, C]
    return o

def spatial_attention(img):

    filters = img.shape[3]
    orig_shape = ((img.shape[1], img.shape[2], img.shape[3]))
    #print(orig_shape)
    img = layers.BatchNormalization()(img)

    # projections:
    q = layers.Conv2D(filters // 8, kernel_size=1, padding="same")(img)
    k = layers.Conv2D(filters // 8, kernel_size=1, padding="same")(img)
    v = layers.Conv2D(filters, kernel_size=1, padding="same")(img)
    k = layers.Reshape((k.shape[1] * k.shape[2], k.shape[3],))(k)

    q = layers.Reshape((q.shape[1] * q.shape[2], q.shape[3]))(q)
    v = layers.Reshape((v.shape[1] * v.shape[2], v.shape[3],))(v)

    img = layers.Lambda(attention)([q, k, v])
    img = layers.Reshape(orig_shape)(img)

    # out_projection:
    img = layers.Conv2D(filters, kernel_size=1, padding="same")(img)
    img = layers.BatchNormalization()(img)

    return img

def sinusoidal_embedding(x, embedding_dims=32):
    embedding_min_frequency = 1.0
    embedding_max_frequency = 1000.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth, use_self_attention=False):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)

            if use_self_attention:
                o = spatial_attention(x)
                x = layers.Add()([x, o])


            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth, use_self_attention=False):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)

            if use_self_attention:
                o = spatial_attention(x)
                x = layers.Add()([x, o])


        return x

    return apply


def get_network(latent_height, latent_width, widths, block_depth, attention_levels, latent_dim=16):
    noisy_latent = keras.Input(shape=(latent_height, latent_width, latent_dim))
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=(latent_height, latent_width), interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_latent)
    x = layers.Concatenate()([x, e])


    skips = []
    level = 0

    for width in widths[:-1]:
        use_self_attention = attention_levels[level]
        x = DownBlock(width, block_depth, use_self_attention)([x, skips, ])
        level += 1
        print(level)

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)
        if bool(attention_levels[level]):
            o = spatial_attention(x)
            x = layers.Add()([x, o])

    for width in reversed(widths[:-1]):
        print(level)
        level -= 1
        use_self_attention = bool(attention_levels[level])
        x = UpBlock(width, block_depth, use_self_attention)([x, skips])

    x = layers.Conv2D(latent_dim, kernel_size=1, kernel_initializer="zeros", activation="linear")(x)

    return keras.Model([noisy_latent, noise_variances], x, name="attention_unet")
