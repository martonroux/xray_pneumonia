import tensorflow as tf
import os
import cv2


def preprocess_data(train_dir: str = '../data/train',
                    test_dir: str = '../data/test',
                    val_dir: str = '../data/val',
                    skip_test=False,
                    img_size=(256, 256)) ->\
        (tf.keras.utils.image_dataset_from_directory,
         tf.keras.utils.image_dataset_from_directory,
         tf.keras.utils.image_dataset_from_directory):

    # TESTING THE DATA TO FIND ANOMALIES

    if skip_test is False:
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

    train_data = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=img_size)
    test_data = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=img_size)
    val_data = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=img_size, batch_size=1)

    # PREPROCESSING

    train_data = train_data.map(lambda x, y: (x / 256, y))
    test_data = test_data.map(lambda x, y: (x / 256, y))
    val_data = val_data.map(lambda x, y: (x / 256, y))

    return train_data, test_data, val_data
