import tensorflow as tf
import os
import cv2


def samplify_data(test_dir: str = 'data/test', train_dir: str = 'data/train', val_dir: str = 'data/val') ->\
        (tf.keras.utils.image_dataset_from_directory,
         tf.keras.utils.image_dataset_from_directory,
         tf.keras.utils.image_dataset_from_directory):

    # TESTING THE DATA TO FIND ANOMALIES

    for image_class in os.listdir(test_dir):
        print(os.path.join(test_dir, image_class))
        for image in os.listdir(os.path.join(test_dir, image_class)):
            image_path = os.path.join(test_dir, image_class, image)
            try:
                cv2.imread(image_path)
            except Exception:
                print("Issue with image {}".format(image_path))

    for image_class in os.listdir(train_dir):
        print(os.path.join(train_dir, image_class))
        for image in os.listdir(os.path.join(train_dir, image_class)):
            image_path = os.path.join(train_dir, image_class, image)
            try:
                cv2.imread(image_path)
            except Exception:
                print("Issue with image {}".format(image_path))

    for image_class in os.listdir(val_dir):
        print(os.path.join(val_dir, image_class))
        for image in os.listdir(os.path.join(val_dir, image_class)):
            image_path = os.path.join(val_dir, image_class, image)
            try:
                cv2.imread(image_path)
            except Exception:
                print("Issue with image {}".format(image_path))

    # TRANSFORMING IMAGES INTO DATASET

    train_data = tf.keras.utils.image_dataset_from_directory(train_dir)
    test_data = tf.keras.utils.image_dataset_from_directory(test_dir)
    val_data = tf.keras.utils.image_dataset_from_directory(val_dir)

    # PREPROCESSING

    train_data = train_data.map(lambda x, y: (x / 255, y))
    test_data = test_data.map(lambda x, y: (x / 255, y))
    val_data = val_data.map(lambda x, y: (x / 255, y))

    return train_data, test_data, val_data
