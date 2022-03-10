import pandas as pd
import os
import time
import numpy as np
np.random.seed(2020)

import keras
import tensorflow as tf
from keras import applications
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

def load_mame(dataframe=False):
    """ Load MAMe dataset data
    Args:
      dataframe (bool): whether to return a dataframe or an array of
                        filenames and a list of labels

    Returns:
      (x_train, y_train), (x_val, y_val), (x_test, y_test) if dataframe=False
      or
      df_train, df_val, df_test if dataframe=True
    """
    INPUT_PATH = 'MAMe_metadata/'

    # Load dataset table
    dataset = pd.read_csv(os.path.join(INPUT_PATH, 'MAMe_dataset.csv'))

    # Subset divisions
    x_train_files = dataset.loc[dataset['Subset'] == 'train']['Image file'].tolist()
    y_train_class = dataset.loc[dataset['Subset'] == 'train']['Medium'].tolist()

    x_val_files = dataset.loc[dataset['Subset'] == 'val']['Image file'].tolist()
    y_val_class = dataset.loc[dataset['Subset'] == 'val']['Medium'].tolist()

    x_test_files = dataset.loc[dataset['Subset'] == 'test']['Image file'].tolist()
    y_test_class = dataset.loc[dataset['Subset'] == 'test']['Medium'].tolist()

    if dataframe:
        train = pd.DataFrame({'filename': x_train_files, 'class': y_train_class})
        val = pd.DataFrame({'filename': x_val_files, 'class': y_val_class})
        test = pd.DataFrame({'filename': x_test_files, 'class': y_test_class})

        # Set full path
        train['filename'] = train['filename'].transform(
            lambda x: INPUT_PATH + 'mame-dataset' + os.sep + 'data' + os.sep + x)
        val['filename'] = val['filename'].transform(
            lambda x: INPUT_PATH + 'mame-dataset' + os.sep + 'data' + os.sep + x)
        test['filename'] = test['filename'].transform(
            lambda x: INPUT_PATH + 'mame-dataset' + os.sep + 'data' + os.sep + x)

        return train, val, test

    else:
        # Return list of filenames
        x_train = [os.path.join(INPUT_PATH, 'mame-dataset', 'data', img_name) for img_name in x_train_files]
        x_val = [os.path.join(INPUT_PATH, 'mame-dataset', 'data', img_name) for img_name in x_val_files]
        x_test = [os.path.join(INPUT_PATH, 'mame-dataset', 'data', img_name) for img_name in x_test_files]

        return (np.array(x_train), np.array(y_train_class)), (np.array(x_val),
                                                              np.array(y_val_class)), (
               np.array(x_test), np.array(y_test_class))


df_train, df_val, df_test = load_mame(dataframe=True)

# Define some variables
img_width, img_height = 256, 256
batch_size = 128
epochs = 100
preprocessing_func = applications.densenet.preprocess_input

# Load dataset
df_train, df_val, df_test = load_mame(dataframe=True)
num_classes = len(df_train['class'].unique())


# Data generation and augmentation

from keras.preprocessing.image import ImageDataGenerator

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
        preprocessing_function = preprocessing_func,
        rotation_range = 30,
        zoom_range = 0.2,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,        # TODO: increase shear - it is in degrees!
        horizontal_flip = True,
        fill_mode = "nearest")

test_datagen = ImageDataGenerator(preprocessing_function = preprocessing_func)

train_generator = train_datagen.flow_from_dataframe(
        df_train,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        class_mode = "categorical",
        validate_filenames=False)

validation_generator = test_datagen.flow_from_dataframe(
        df_val,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        shuffle = False,
        class_mode = "categorical",
        validate_filenames=False)

test_generator = test_datagen.flow_from_dataframe(
        df_test,
        target_size = (img_height, img_width),
        batch_size = 1,
        shuffle = False,
        class_mode = "categorical",
        validate_filenames=False)
