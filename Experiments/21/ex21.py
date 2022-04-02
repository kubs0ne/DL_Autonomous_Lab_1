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
import numpy as np

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)
import DataGenerator
import ModelEvaluator

img_width, img_height = 256, 256
batch_size = 32
epochs = 60

train_generator, validation_generator, test_generator = DataGenerator.data_Gens(parentparentdir, img_height, img_width,
                                                                                batch_size)
def custom_gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
get_custom_objects().update({'custom_gelu': Activation(custom_gelu)})

# Define the NN architecture
weight_decay = 1e-4

input = Input(shape = (img_width, img_height, 3))
x = Conv2D(32, (3, 3), padding='same')(input)
x = Activation(custom_gelu)(x)
x = BatchNormalization()(x)

a1 = Conv2D(32, (3, 3), padding='same')(x)
a1 = Activation(custom_gelu)(a1)
a1 = BatchNormalization()(a1)
a1 = MaxPooling2D(pool_size=(2, 2))(a1)

a2 = Conv2D(64, (3, 3), padding='same')(a1)
a2 = Activation(custom_gelu)(a2)
a2 = BatchNormalization()(a2)
a2 = MaxPooling2D(pool_size=(2, 2))(a2)

b1 = Conv2D(64, (5, 5), padding='same')(x)
b1 = Activation(custom_gelu)(b1)
b1 = BatchNormalization()(b1)
b1 = MaxPooling2D(pool_size=(2, 2))(b1)


b2 = Conv2D(128, (7, 7), padding='same')(b1)
b2 = Activation(custom_gelu)(b2)
b2 = BatchNormalization()(b2)
b2 = MaxPooling2D(pool_size=(2, 2))(b2)

a1b1 = Concatenate()([a1, b1])

c1 = (Conv2D(128, (3, 3), padding='same'))(a1b1)
c1 = Activation(custom_gelu)(c1)
c1 = BatchNormalization()(c1)
c1 = MaxPooling2D(pool_size=(2, 2))(c1)

a2b2 = Concatenate()([a2, b2])

c2 = (Conv2D(128, (3, 3), padding='same'))(a2b2)
c2 = Activation(custom_gelu)(c2)
c2 = BatchNormalization()(c2)
# c2 = MaxPooling2D(pool_size=(2, 2))(c2)

c1c2 = Concatenate()([c1, c2])

d = (Conv2D(48, (3, 3), padding='same'))(c1c2)
d = Activation(custom_gelu)(d)
d = BatchNormalization()(d)

x = GlobalAveragePooling2D()(d)
x = Flatten()(x)

x = Dense(29, activation=(tf.nn.softmax))(x)

model = Model(inputs = input, outputs = x)

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
