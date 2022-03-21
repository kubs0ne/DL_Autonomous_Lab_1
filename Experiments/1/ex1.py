import os
import sys
import inspect
import time
import tensorflow as tf

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)
import DataGenerator
import ModelEvaluator


img_width, img_height = 256, 256
batch_size = 128
epochs = 100

train_generator, validation_generator, test_generator = DataGenerator.data_Gens(parentparentdir, img_height, img_width, batch_size)

print(test_generator.classes)

#Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
#Two hidden layers
model = Sequential()
model.add(Conv2D(96, (5, 5), activation='relu', input_shape=(img_width, img_height, 3),  padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(80, (5, 5), activation='relu', input_shape=(img_width, img_height, 3),  padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(96, 5, 5, activation='relu', padding="same"))
model.add(Conv2D(64,5, 5, activation='relu', padding="same"))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(29, activation=(tf.nn.softmax)))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# Train the model
t0 = time.time()
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VAL = validation_generator.n // validation_generator.batch_size

early = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10, verbose=1, mode='auto',
                      restore_best_weights=True)

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch= 1, #STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps= 1, #STEP_SIZE_VAL,
    epochs=1
)

print('Model trained in {:.1f}min'.format((time.time() - t0) / 60))

ModelEvaluator.evaluate_model(model,history,  validation_generator)
