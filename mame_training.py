import pandas as pd
import os
import time
import numpy as np
np.random.seed(2020)
import PIL

import keras
import tensorflow as tf
from keras import applications
from tensorflow.keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.applications import densenet

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
print(df_train.head())

# Loading pre-trained model

print('Using Keras version', keras.__version__)
print('Using TensorFlow version', tf.__version__)

# Define some variables
img_width, img_height = 256, 256
batch_size = 128
epochs = 100
preprocessing_func = densenet.preprocess_input

# Load dataset
df_train, df_val, df_test = load_mame(dataframe=True)
num_classes = len(df_train['class'].unique())

# Print information about loaded data
print('Training examples: {}\n{}\n'.format(len(df_train), df_train.head()))
print('Validation examples: {}\n{}\n'.format(len(df_val), df_val.head()))

# Load pre-trained model
base_model = densenet.DenseNet121(weights="imagenet", include_top=False,
                                      input_shape=(img_width, img_height, 3))  # pooling = max/avg

# Freeze layers
# Train all layers after first that are not a BatchNormalization layer
first_to_train = 'conv5_block14_1_conv'
base_model.trainable = True
set_trainable = False
for layer in base_model.layers:
    if layer.name == first_to_train:
        set_trainable = True

    if set_trainable and not isinstance(layer, BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False

print('Number of initial layers: {}'.format(len(base_model.layers)))

# Adding custom Layers
x = base_model.output
x = GlobalMaxPooling2D()(x)  # less features than Flatten
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation="softmax", name='Predictions')(x)

# Creating the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.001, epsilon=0.1, amsgrad=True),
              metrics=["accuracy"])

# Print model summary
print('Number of final layers: {}'.format(len(model.layers)))
print(model.summary())

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

# Training

# Save the model according to the conditions
# checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10, verbose=1, mode='auto',
                      restore_best_weights=True)

# Train the model
t0 = time.time()
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VAL = validation_generator.n // validation_generator.batch_size

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VAL,
    epochs=epochs,
    use_multiprocessing=True,
    workers=6,
    callbacks=[early]
)

print('Model trained in {:.1f}min'.format((time.time() - t0) / 60))