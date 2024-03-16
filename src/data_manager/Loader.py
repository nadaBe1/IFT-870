import os
import pandas as pd

class Loader():
    """
    A class for loading datasets from a specified path.
    """

    def __init__(self, path):
        """
        Initializes the Loader instance with the specified path.
        """
        self.path = path
        self.trainset = None
        self.testset = None

    def load(self):
        """
        Loads the training and test datasets from the specified path.
        """
        self.trainset = pd.read_csv(os.path.join(self.path, '../data/train.csv'))
        self.testset = pd.read_csv(os.path.join(self.path, '../data/test.csv'))

    def get_trainset(self):
        """
        Returns the training dataset.
        """
        return self.trainset

    def get_testset(self):
        """
        Returns the test dataset.
        """
        return self.testset