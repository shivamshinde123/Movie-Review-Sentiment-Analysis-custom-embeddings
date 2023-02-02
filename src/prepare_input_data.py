import logging, os, yaml
import string, re
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding
from get_data import GetData
from reading_params import ReadParams


## creating a logger, file handler and the formatter
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

params = ReadParams().read_params()

data_encoding_log_file_path = params['Log_paths']['data_encoding']

file_handler = logging.FileHandler(data_encoding_log_file_path)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class DataEncoding:

    """The purpose of this class and its methods is to encode the text data

    Parameters
    -----------

    sequence_length : int
        Length of biggest sentence (default 300)
    vocab_size: int
        The size of vovabulary (default 20000)
    embedding_dim: int
        The size of output of embedding layer
    
    """

    def __init__(self, sequence_length=300, vocab_size=20000, embedding_dim=300):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    @tf.keras.utils.register_keras_serializable(package='Custom', name=None)
    def custom_standardization(self, input_data):

        """The purpose of this function is to perform a cleaning operation on the text data before its encoding.

        Parameters
        -----------

        input_data: text document
            All the cleaning operation will be performed on this input data

        Returns
        --------

        tokens: input data after cleaning operation
        """

        try:
            # removing the punctuations
            tokens = tf.strings.regex_replace(input_data, '[%s]' % re.escape(string.punctuation), '')

            logger.info('Punctuations removed from the input data.')
            
            # removing un-printable characters
            tokens = tf.strings.regex_replace(tokens, '[^%s]' % re.escape(string.printable), '')

            logger.info('Unprintable characters removed from the input data.')
            
            # removing english stopwords
            stop_words = set(stopwords.words('english'))
            for word in stop_words:
                tokens = tf.strings.regex_replace(tokens, f'{word}', "")

            logger.info('English language stopwords removed from the input data.')
            
            ## lowering the case of all the text
            tokens = tf.strings.lower(tokens)

            logger.info('All the characters from the input data converted to the lower case.')
            
        except Exception as e:
            logger.exception(e)
            raise e
        
        else:
            return tokens

    def create_vectorization_layer(self):

        """This function is used to integer encode the input data

        Parameters
        -----------

        None

        Returns
        --------

        vectorize_layer: tensorflow layer for integer encoding of text input data
        """

        try:
            vectorize_layer = TextVectorization(
            standardize = self.custom_standardization,
            max_tokens = self.vocab_size,
            output_mode = 'int',
            output_sequence_length = self.sequence_length)

            logger.info('TextVectorization layer created for the integer encoding of input text data.')

            train_ds, val_ds = GetData().get_train_and_val_data()

            text_ds = train_ds.map(lambda x, y: x)

            vectorize_layer.adapt(text_ds)

            logger.info('TextVectorization layer adapted on the training data.')

        except Exception as e:
            logger.exception(e)
            raise e
        
        else:
            return vectorize_layer

    
    def create_embedding_layer(self):

        """This function is used to create the embeddings of the input integer encoded text

        Parameters
        -----------

        None

        Returns
        --------

        text_embedding: embedded vector for the input integer encoded data
        """

        try:
            text_embedding = Embedding(self.vocab_size, self.embedding_dim, name='embedding')

            logger.info('Created embeddings for the input integer encoded data.')

        except Exception as e:
            logger.exception(e)
            raise e

        else:
            return text_embedding


    


        


    
