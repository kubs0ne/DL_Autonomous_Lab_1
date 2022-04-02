import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def evaluate_model(model, history, eval_gen ):
    """ Evaluate given model and print results.
    Show validation loss and accuracy, classification report and
    confusion matrix.

    Args:
        model (model): model to evaluate

        eval_gen (ImageDataGenerator): evaluation generator
    """
    # Evaluate the model
    eval_gen.reset()
    score = model.evaluate(eval_gen, verbose=0)
    print('validation loss:', score[0])
    print('validation accuracy:', score[1])

    # Confusion Matrix (validation subset)
    eval_gen.reset()
    pred = model.predict(eval_gen, verbose=0)
    predicted_class_indices = np.argmax(pred, axis=1)    # Assign most probable label
    labels = (eval_gen.class_indices)
    target_names = labels.keys()

    print(classification_report(eval_gen.classes, predicted_class_indices, target_names=target_names))

    

    #Accuracy plot
    acc_plot(history)
    #Loss plot
    loss_plot(history)

def conf_matrix(c):
    cf_matrix = confusion_matrix(np.array(eval_gen.classes), predicted_class_indices)
    fig, ax = plt.subplots(figsize=(13, 13))
    sns.heatmap(cf_matrix, annot=True, cmap='PuRd', cbar=False, square=True, xticklabels=target_names,
                yticklabels=target_names)
    plt.show()
    plt.savefig('confusion_matrix.pdf')
    plt.close()

def clf_report():
    with open('Classification_Report.txt', 'w') as f:
        f.write('Create a new text file!')

def acc_plot(history):
    matplotlib.use('Agg')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig('accuracy.pdf')
    plt.close()

def loss_plot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig('loss.pdf')