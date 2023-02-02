import json, yaml, os
from get_data import GetData
import logging

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from prepare_input_data import DataEncoding
from reading_params import ReadParams


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

params = ReadParams().read_params()

model_evaluation_log_file_path = params['Log_paths']['model_evaluation']

file_handler = logging.FileHandler(model_evaluation_log_file_path)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class EvaluateTrainedModel:


    """This class is used to evaluate the performance of trained sentiment analysis model on testing data."""


    def __init__(self):
        pass

    def evalute_model(self):

        """This method is used to evalute a trained sentiment analysis model over the test data.

        Parameters
        -----------

        None

        Returns
        --------

        None
        """

        try:
            if not os.path.exists('Logs'):
                os.makedirs('Logs')
                
            params = ReadParams().read_params()
            model_filepath = params['Model_paths']['model_path']
            loaded_model = None
            # with custom_object_scope({'custom_standardization': DataEncoding().custom_standardization}):
            loaded_model = load_model(model_filepath)

            logger.info('Trained sentiment analysis model loaded.')

            test_ds = GetData().get_test_data()

            logger.info('Test data loaded.')

            test_data_loss, test_data_accuracy = loaded_model.evaluate(test_ds)

            logger.info('Loss and accuracy of trained sentiment analysis over the test data calculated.')

            params = ReadParams().read_params()

            evaluation_metrics_file_path = params['Metric_paths']['model_evaluation']
            with open(evaluation_metrics_file_path, 'w') as json_file:
                test_data_evaluation = {
                    'test_data_loss': test_data_loss,
                    'test_data_accuracy': test_data_accuracy
                }

                json.dump(test_data_evaluation, json_file, indent=4)

            logger.info('Loss and accuracy of trained sentiment analysis over the test data saved to the json file.')

        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == "__main__":

    eval = EvaluateTrainedModel()
    eval.evalute_model()


        