## Importing required libraries
import logging, os
import tensorflow as tf
from tensorflow.keras.utils import text_dataset_from_directory
from reading_params import ReadParams


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

params = ReadParams().read_params()
data_loading_log_file_path = params['Log_paths']['data_loading']

file_handler = logging.FileHandler(data_loading_log_file_path)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class GetData:

    """Class for the purpose of loading the data

    Parameters
    -----------

    batch_size :  int
            batch size (default 1024)
    validation split : float value between 0 and 1
            validation split ratio used while creating training and validation data (default 0.2)
    seed : int
            random seed value (default 23)
    train_path: str
            path of the training data (default Raw_data/Data/train)
    """

    def __init__(self, batch_size=1024, validation_split=0.2, seed=23):
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.seed = seed

    def get_train_and_val_data(self):


        """Create a tensorflow data object for training and validation data.

        Parameters
        -----------

        None

        Returns
        ---------

        train_ds : tensorflow data object
                tensorflow data object for training
        val_ds: tensorflow data object
                tensorflow data object for validation

        """

        try:
            params = ReadParams().read_params()

            train_data_path = params['Data_paths']['train_data_path']

            train_ds = text_dataset_from_directory(train_data_path,
                                        batch_size = self.batch_size,
                                        validation_split = self.validation_split,
                                        subset='training',
                                        seed = self.seed)

            logger.info('Training Data Loaded.')

            class_names_file_path = params['Metric_paths']['class_names']

            with open(class_names_file_path, 'w') as f:
                    f.write(f"Label 0 corresponds to {train_ds.class_names[0]} and label 1 corresponds to {train_ds.class_names[1]}.")

            logger.info('Class names checked.')

            test_data_path = params['Data_paths']['test_data_path']
            val_ds = text_dataset_from_directory(test_data_path,
                                            batch_size = self.batch_size,
                                            validation_split = self.validation_split,
                                            subset='validation',
                                            seed = self.seed)

            logger.info('Validation Data Loaded.')

        except Exception as e:
            logger.exception(e)
            raise e

        else:
            return train_ds, val_ds

    
    def get_test_data(self):

        """Create a tensorflow data object for testing data.

        Parameters
        -----------

        None

        Returns
        ---------

        test_ds : tensorflow data object
                tensorflow data object for testing
    
        """

        try:

            params = ReadParams().read_params()

            test_path = params['Data_paths']['test_data_path']
            
            test_ds = text_dataset_from_directory(
            test_path,
            batch_size = self.batch_size)

            logger.info('Testing Data Loaded.')

        except Exception as e:
            logger.exception(e)
            raise e
        
        else:
            return test_ds

    
    
