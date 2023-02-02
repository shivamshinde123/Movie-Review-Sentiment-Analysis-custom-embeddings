import yaml

class ReadParams:

    """This class is used to read the variables stored in parameters yaml file"""

    def __init__(self, params_path='params.yaml'):
        self.params_path = params_path

    def read_params(self):

        """This method is used to read the parameters yaml file and returns the loaded file object.
        
        Parameters
        -----------
            
        config_path: Path to the parameters yaml file
        
        Returns
        --------

        Loaded yaml file object: Returns the yaml file object.
        """

        try:
            with open(self.params_path) as params_file:
                params = yaml.safe_load(params_file)
            
        except Exception as e:
            raise e

        else:
            return params