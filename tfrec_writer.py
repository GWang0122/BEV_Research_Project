import os
import tensorflow as tf
import random

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, tf.Tensor):
        value = value.numpy()  # Convert the tensor to a numpy array.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_image_paths(root_dir, target_layer, train_ratio=0.9):
    image_paths = []
    for scene_folder in os.listdir(root_dir):
        scene_folder_path = os.path.join(root_dir, scene_folder)

        for numbered_folder in os.listdir(scene_folder_path):
            numbered_folder_path = os.path.join(scene_folder_path, numbered_folder)

            target_layer_file = os.path.join(numbered_folder_path, target_layer)
            if os.path.exists(target_layer_file):
                image_paths.append(target_layer_file)

    # Shuffle the list of image paths
    random.shuffle(image_paths)

    # Calculate the split index
    split_index = int(len(image_paths) * train_ratio)

    # Split the image paths into train and validation sets
    train_image_paths = image_paths[:split_index]
    valid_image_paths = image_paths[split_index:]

    return train_image_paths, valid_image_paths

def serialize(image, path):
    feature = {
        "image": _bytes_feature(tf.io.encode_png(image)),
        "path": _bytes_feature(path.encode()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))
'''
def write_to_tfrecord(paths, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for path in paths:
            with open(path, 'rb') as file:
                image_string = file.read()

            # Check if the image_string is not empty before processing
            if image_string:
                image = tf.image.decode_png(image_string, channels=1)
                example = serialize(image, path)
                writer.write(example.SerializeToString())
'''
def write_jpg_to_tfrecord(paths, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for path in paths:
            with open(path, 'rb') as file:
                image_string = file.read()

            # Check if the image_string is not empty before processing
            if image_string:
                image = tf.image.decode_jpeg(image_string, channels=3)
                example = serialize(image, path)
                writer.write(example.SerializeToString())

def parse_tfrec(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_png(example["image"], channels=3)
    path = example["path"]

    return image, path


def parse_images(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_png(example["image"], channels=3)
    image = tf.cast(image, tf.float32) / 255.0

    return image

def main():
    root_dir = "../alignment_generator/NEW_PROCESSED_DATA"
    train_paths, valid_paths = get_image_paths(root_dir , "combined_rgb.jpg")
    write_jpg_to_tfrecord(train_paths, "train_data.tfrecord")
    write_jpg_to_tfrecord(valid_paths, "valid_data.tfrecord")

if __name__ == "__main__":
    main()



