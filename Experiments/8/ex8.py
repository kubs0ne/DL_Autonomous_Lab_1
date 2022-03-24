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
batch_size = 64
epochs = 30

train_generator, validation_generator, test_generator = DataGenerator.data_Gens(parentparentdir, img_height, img_width, batch_size)


#Define the NN architecture

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D
#Two hidden layers
model= tf.keras.applications.DenseNet121(include_top=True,
                   input_shape=(img_width,img_width,3),classes=29,
                   weights=None)
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
    steps_per_epoch= STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps= STEP_SIZE_VAL,
    epochs=epochs
)

print('Model trained in {:.1f}min'.format((time.time() - t0) / 60))

ModelEvaluator.evaluate_model(model, history, validation_generator)
