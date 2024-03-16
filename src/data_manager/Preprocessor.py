import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir) 

from analyzer.Analyzer import Analyzer

class Preprocessor():

    def drop_column(self, dataset, col):
        """
        Function that drops the specified column of the specified dataset.
        """
        dataset = dataset.drop(columns=col, axis=1)
        return dataset

    def replace_outliers_na(self, dataset, columns, threshold=1.5):
        """
        Function that replaces outliers by nan values using lower bound and upper bound strategy.
        """
        for column in columns:
            # Calculate IQR for the column
            Q1 = dataset[column].quantile(0.25)
            Q3 = dataset[column].quantile(0.75)
            IQR = Q3 - Q1
            # Define lower and upper bounds for outliers
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            # Replace outliers from the column by nan values
            dataset[column] = np.where(
                (dataset[column] <= lower_bound) | (dataset[column] >= upper_bound),
                np.nan,
                dataset[column]
            )
        return dataset

    def drop_column_na(self, dataset, percentage):
        """
        Function that drops columns that contain more than the specified percentage of nan values.
        """
        column_percentage = Analyzer.percentage_na_by_column(Analyzer(), dataset)
        column_to_drop = dataset.columns[np.where(column_percentage > percentage)]
        dataset = self.drop_column(dataset, column_to_drop)
        return dataset

    def knn_imputer(self, dataset, column):
        """
        Imputation for completing missing values using KNNImputer from scikit-learn (k-Nearest Neighbors).
        """
        imputer = KNNImputer()
        dataset[column] = imputer.fit_transform(dataset[column])
        return dataset

    def encoding(self, dataset, column):
        """
        Function that encodes the specified column of the specified dataset.
        """
        dataset_encoded = dataset.copy()
        encoder = LabelEncoder()
        dataset_encoded[column] = encoder.fit_transform(dataset_encoded[column])
        return dataset_encoded

    def standardization(self, dataset, column):
        """
        Function that standardizes the numeric columns of the specified dataset.
        """
        dataset_standardized = dataset.copy()
        scaler = StandardScaler()
        dataset_standardized[column] = scaler.fit_transform(dataset_standardized[column])
        return dataset_standardized

    def shuffle(self, dataset):
        """
        Function that shuffles the values of the specified dataset.
        """
        dataset = shuffle(dataset)
        return dataset

    def split_dataset(self, dataset, target):
        """
        Function that splits the dataset into random train and test subsets using train_test_split from Scikit Learn.
        """
        X = dataset.drop(target, axis=1)
        y = dataset[target]
        return X, y
    
    def rfe_selection(self, dataset, features, target, estimator):
        """
        Function that applies the recursive feature elimination algorithm using RFE from Scikit Learn.
        """
        selector = RFE(estimator, step=1)
        selector = selector.fit(features, target)
        columns = list(selector.get_feature_names_out())+[target.name]
        return dataset[columns]
    
    def pca_reduction(self, dataset, target, nb_comp):
        """
        Function that applies the Principal component analysis algorithm using PCA from Scikit Learn.
        """
        pca = PCA(n_components=nb_comp)
        X_pca = pca.fit_transform(dataset)
        X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(nb_comp)])
        X_pca_df[target.name] = target.values
        return pca, X_pca_df