import os
import sys
import inspect
import time
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)
import DataGenerator
import ModelEvaluator

df_train, df_val, df_test = DataGenerator.load_mame(parentparentdir,dataframe=True)

img_width, img_height = 256, 256
batch_size = 128
epochs = 100
# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
        # preprocessing_function = preprocessing_func,
        rotation_range = 30,
        zoom_range = 0.2,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,        # TODO: increase shear - it is in degrees!
        horizontal_flip = True,
        fill_mode = "nearest")

test_datagen = ImageDataGenerator(
    # preprocessing_function = preprocessing_func
    )

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
    steps_per_epoch=1,
    validation_data=validation_generator,
    validation_steps=1,
    epochs=1
)

print('Model trained in {:.1f}min'.format((time.time() - t0) / 60))

ModelEvaluator.evaluate_model(model,history,  validation_generator)
