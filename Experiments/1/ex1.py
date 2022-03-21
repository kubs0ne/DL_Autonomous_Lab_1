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
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VAL,
    epochs=1
)

print('Model trained in {:.1f}min'.format((time.time() - t0) / 60))

ModelEvaluator.evaluate_model(model,history,  validation_generator)


""" 
#Evaluate the model with test set
#score = model.evaluate(test_generator, verbose=0)
#print('test loss:', score[0])
#print('test accuracy:', score[1])

##Store Plots
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#Accuracy plot
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train','val'], loc='upper left')
#plt.savefig('Results/mnist_fnn_accuracy.pdf')
#plt.close()
#Loss plot
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train','val'], loc='upper left')
#plt.savefig('Results/mnist_fnn_loss.pdf')

import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
"""

#def evaluate_model(model, eval_gen):
    """ Evaluate given model and print results.
    Show validation loss and accuracy, classification report and
    confusion matrix.

    Args:
        model (model): model to evaluate
        eval_gen (ImageDataGenerator): evaluation generator
    """
    # Evaluate the model
    """
    eval_gen.reset()
    score = model.evaluate(eval_gen, verbose=0)
    print('\nLoss:', score[0])
    print('Accuracy:', score[1])
    # Confusion Matrix (validation subset)
    eval_gen.reset()
    pred = model.predict(eval_gen, verbose=0)

    # Assign most probable label
    predicted_class_indices = np.argmax(pred, axis=1)

    # Get class labels
    labels = (eval_gen.class_indices)
    target_names = labels.keys()

   # Plot statistics
    print(classification_report(eval_gen.classes, predicted_class_indices, target_names=target_names))

    cf_matrix = confusion_matrix(np.array(eval_gen.classes), predicted_class_indices)
    fig, ax = plt.subplots(figsize=(13, 13))
    sns.heatmap(cf_matrix, annot=True, cmap='PuRd', cbar=False, square=True, xticklabels=target_names,
                yticklabels=target_names)
    plt.show()
    plt.savefig('Results/ex1.pdf')

 """