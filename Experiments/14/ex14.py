import os
import sys
import inspect
import time
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Input
from keras import regularizers
from keras.utils.generic_utils import get_custom_objects
from keras.backend import int_shape

import numpy as np

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)
import DataGenerator
import ModelEvaluator

img_width, img_height = 256, 256
batch_size = 64
epochs = 30

train_generator, validation_generator, test_generator = DataGenerator.data_Gens(parentparentdir, img_height, img_width,
                                                                                batch_size)

def custom_gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
get_custom_objects().update({'custom_gelu': Activation(custom_gelu)})


##### BerNet

input = Input(shape = (img_width, img_height, 3))

l1 = Sequential([
    Conv2D(32, (3, 3), padding='same'),
    Activation(custom_gelu), 
    BatchNormalization()])

l2 = Sequential([
    Conv2D(32, (3, 3), padding='same'),
    Activation(custom_gelu), 
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2))])

l3 = Sequential([
    Conv2D(64, (3, 3), padding='same'),
    Activation(custom_gelu), 
    BatchNormalization()])

l4 = Sequential([
    Conv2D(64, (3, 3), padding='same'),
    Activation(custom_gelu), 
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2))])

l5 = Sequential([
    Conv2D(128, (3, 3), padding='same'),
    Activation(custom_gelu), 
    BatchNormalization()])

l6 = Sequential([
    Conv2D(128, (3, 3), padding='same'),
    Activation(custom_gelu), 
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2))])


classifier = Sequential([
    Flatten(),
    Dense(29, activation=(tf.nn.softmax))])

x1 = l1(input)
x2 = l2(x1)
red_x1 = MaxPooling2D(pool_size=(2,2))(x1)
x12 = Concatenate()([red_x1, x2]) 
x3 = l3(x12)
x4 = l4(x3)
red_x12 = MaxPooling2D(pool_size=(2,2))(x12)
x124 = Concatenate()([red_x12, x4]) 
x5 = l5(x124)
x6 = l6(x5)
red_x124 = MaxPooling2D(pool_size=(2,2))(x124)
x7 = Concatenate()([red_x124, x6]) 
output = classifier(x7)

model = Model(inputs = input, outputs = output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
tf.keras.utils.plot_model(model)


# Train the model
t0 = time.time()
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VAL = validation_generator.n // validation_generator.batch_size

early = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10, verbose=1, mode='auto',
                      restore_best_weights=True)

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VAL,
    epochs=epochs
)

model.save('model')

print('Model trained in {:.1f}min'.format((time.time() - t0) / 60))

ModelEvaluator.evaluate_model(model, history, validation_generator)
