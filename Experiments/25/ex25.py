import os
import sys
import inspect
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)
import DataGenerator
import ModelEvaluator
from keras.utils.generic_utils import get_custom_objects
import numpy as np

img_width, img_height = 256, 256
batch_size = 64
epochs = 50

train_generator, validation_generator, test_generator = DataGenerator.data_Gens(parentparentdir, img_height, img_width, batch_size)

def custom_gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
get_custom_objects().update({'custom_gelu': Activation(custom_gelu)})


# Define the NN architecture
weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', input_shape=(img_width, img_height, 3)))
model.add(Activation(custom_gelu))
model.add(BatchNormalization())

model.add(Conv2D(32, (3,3), padding='same'))
model.add(Activation(custom_gelu))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
 
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation(custom_gelu))
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation(custom_gelu))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
 
model.add(Conv2D(128, (3,3), padding='same'))
model.add(Activation(custom_gelu))
model.add(BatchNormalization())

model.add(Conv2D(128, (3,3), padding='same'))
model.add(Activation(custom_gelu))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3), padding='same'))
model.add(Activation(custom_gelu))
model.add(BatchNormalization())

model.add(Conv2D(256, (3,3), padding='same'))
model.add(Activation(custom_gelu))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(4,4)))


model.add(GlobalAveragePooling2D())
model.add(Flatten())

model.add(Dense(29, activation=(tf.nn.softmax)))
 
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# Train the model
t0 = time.time()
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VAL = validation_generator.n // validation_generator.batch_size

early = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10, verbose=1, mode='auto',
                      restore_best_weights=True)

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch= STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps= STEP_SIZE_VAL,
    epochs=epochs
)
model.save('model')
print('Model trained in {:.1f}min'.format((time.time() - t0) / 60))

ModelEvaluator.evaluate_model(model, history, validation_generator)
