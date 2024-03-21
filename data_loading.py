import os
import cv2
import math


def open_preprocess_photos(
        dir: str,  # The directory in which to search for images
        transform,  # The pytorch Transform function to apply to images
        img_size: tuple  # The wanted output image size
) -> list:
    img_list = list(map(lambda x: os.path.join(dir, x), os.listdir(dir)))  # Open given directory and list its contents
    img_list.sort()
    images = []  # This list will store the preprocessed images to return

    # Preprocess from pre_process_1.ipynb
    for img in img_list:
        try:
            img_data = cv2.imread(img)
        except Exception:
            print(f"Error occured opening {img} file")
            continue
        shape = img_data.shape

        if shape[0] > shape[1]:  # If axis 0 is bigger than axis 1, we crop the image according to axis 0
            diff = shape[0] - shape[1]
            padding = int(math.ceil(diff / 2))
            # We center the image according to axis 0, to make it square
            img_data = img_data[padding: shape[0] - padding, :, :]
        else:  # If axis 1 is bigger than axis 0, we crop the image according to axis 1
            diff = shape[1] - shape[0]
            padding = int(math.ceil(diff / 2))
            # We center the image according to axis 1, to make it square
            img_data = img_data[:, padding: shape[1] - padding, :]

        img_data = cv2.resize(img_data, img_size)  # Resize image according to given size ((224, 224), for example)
        img_data = transform(img_data)  # Transform the image using the Pytorch transform function given as parameter
        images.append(img_data)

    return images


def open_preprocess_photos_flip(
        dir: str,  # The directory in which to search for images
        transform,  # The pytorch Transform function to apply to images
        img_size: tuple  # The wanted output image size
) -> list:
    img_list = list(map(lambda x: os.path.join(dir, x), os.listdir(dir)))  # Open given directory and list its contents
    img_list.sort()
    images = []  # This list will store the preprocessed images to return

    # Preprocess from pre_process_1.ipynb
    for img in img_list:
        try:
            img_data = cv2.imread(img)
        except Exception:
            print(f"Error occurred opening {img} file")
            continue
        shape = img_data.shape

        if shape[0] > shape[1]:  # If axis 0 is bigger than axis 1, we crop the image according to axis 0
            diff = shape[0] - shape[1]
            padding = int(math.ceil(diff / 2))
            # We center the image according to axis 0, to make it square
            img_data = img_data[padding: shape[0] - padding, :, :]
        else:  # If axis 1 is bigger than axis 0, we crop the image according to axis 1
            diff = shape[1] - shape[0]
            padding = int(math.ceil(diff / 2))
            # We center the image according to axis 1, to make it square
            img_data = img_data[:, padding: shape[1] - padding, :]

        img_data = cv2.resize(img_data, img_size)  # Resize image according to given size ((224, 224), for example)
        flipped = cv2.flip(img_data, 1)  # Flip the image horizontally to do data augmentation
        images.append(transform(img_data))
        images.append(transform(flipped))

    return images
