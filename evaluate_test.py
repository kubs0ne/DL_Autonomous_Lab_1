import tensorflow as tf
import sys
import DataGenerator
import ModelEvaluator

num = sys.argv[1]
model = tf.keras.models.load_model(f'Experiments/{num}/model')

train_generator, validation_generator, test_generator = DataGenerator.data_Gens(parentparentdir, img_height, img_width,
                                                                                batch_size)

ModelEvaluator.evaluate_model(model, validation_generator)