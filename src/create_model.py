import logging, yaml, os
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling1D
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model

from prepare_input_data import DataEncoding
from get_data import GetData
from reading_params import ReadParams


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

params = ReadParams().read_params()

model_creation_log_file_path = params['Log_paths']['model_creation']
  
file_handler = logging.FileHandler(model_creation_log_file_path)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class ModelCreation:

    """This function is used to create a deep learning model for sentiment analysis

    Parameters
    -----------
    
    None
    """

    def __init__(self):
        pass

    def create_input_layer(self):

        """This function is used to create a input layer for the model

        Parameters
        -----------

        None

        Returns
        --------

        text_input : tensorflow layer
            This is the input layer of the model
        """

        try:
            vectorize_layer = DataEncoding().create_vectorization_layer()
            text_embedding = DataEncoding().create_embedding_layer()

            text_input = Sequential(
                [vectorize_layer, text_embedding], name = 'cleaned_encoded_input'
            )

            logger.info('Input layer created.')

        except Exception as e:
            logger.exception(e)
            raise e
        
        else:
            return text_input


    def create_classifier(self):

        """This function will create a layers for classification

        Parameters
        -----------

        None

        Returns
        ---------
        classifier_head: group of layers for the classification part of model
        """

        try:
            classifier_head = Sequential(
             [GlobalAveragePooling1D(), 
             Dense(16, activation='relu'), 
             Dense(32, activation='relu'), 
             Dense(16, activation='relu'), 
             Dense(1, activation='sigmoid')],
            name = 'classifier_head')

            logger.info('Classifier head created.')

        
        except Exception as e:
            logger.exception(e)
            raise e

        else:
            return classifier_head


    def create_classification_model(self):

        """This function is used to create a whole classification model using the input layer and classification layers

        Parameters
        -----------

        None

        Returns
        --------

        model : tensorflow sequential model for sentiment analysis
        """

        try:
            if not os.path.exists('Logs'):
                os.makedirs('Logs')
            if not os.path.exists('Metrics'):
                os.makedirs('Metrics')
            if not os.path.exists('Plots'):
                os.makedirs('Plots')
            if not os.path.exists('Models'):
                os.makedirs('Models')
            ## getting the text_input and classifier_head
            text_input = self.create_input_layer()
            classifier_head = self.create_classifier()

            ## creating a model
            model = Sequential([text_input, classifier_head])

            logger.info('Tensorflow sequential model for sentiment analysis created.')

            ## compiling a model
            model.compile(
                optimizer="adam",
                loss=BinaryCrossentropy(from_logits=True),
                metrics=["accuracy"])

            logger.info('Tensorflow sequential model for sentiment analysis compiled.')

            ## saving a model summary

            params = ReadParams().read_params()

            model_summary_file_path = params['Metric_paths']['model_summary']

            with open(model_summary_file_path, 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))

            logger.info('Model summary saved.')

            ## creating a early stopping callback
            early_stopping = EarlyStopping(restore_best_weights=True, patience=5)


            train_ds, val_ds  = GetData().get_train_and_val_data()

            ## training a model
            history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=15,
                        callbacks=[early_stopping])
            ## saving the training procedure to the csv file

            train_procedure_file_path = params['Metric_paths']['training_procedure']
            pd.DataFrame(history.history, columns=['loss', 'accuracy', 'val_loss', 'val_accuracy']).to_csv(train_procedure_file_path, index=False)

            logger.info('Sentiment analysis model trained.')

            ## saving a trained model to a pickle file
            model_filepath = params['Model_paths']['model_path']
            model.save(model_filepath)
            # save_model(model = model, filepath = model_filepath, save_format='h5')

            logger.info('Sentiment analysis trained model saved to a pickle file.')

            ## plotting and saving a plot for training procedure
            plt.style.use('seaborn')
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,8))
            fig.suptitle('Training over epochs', fontsize=16)

            ax1.plot(history.history['loss'],color='blue',label='Training loss')
            ax1.plot(history.history['val_loss'],color='red',label='Validation loss')
            ax1.legend()
            ax1.set_ylabel('Loss')
            ax1.set_xlabel('Epochs')

            ax2.plot(history.history['accuracy'],color='blue',label='Training accuracy')
            ax2.plot(history.history['val_accuracy'],color='red',label='Validation accuracy')
            ax2.legend()
            ax2.set_ylabel('Accuracy')
            ax2.set_xlabel('Epochs')

            plt.tight_layout()

            plot_filepath = params['Plot_paths']['training_procedure']
            plt.savefig(plot_filepath)

            logger.info('Training over epochs plot saved.')
            plt.close(fig)
        
        except Exception as e:
            logger.exception(e)
            raise e



if __name__ == "__main__":

    model_creation = ModelCreation()
    model_creation.create_classification_model()