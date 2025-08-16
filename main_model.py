import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NLPModelException(Exception):
    """Base exception class for NLP model"""
    pass

class InvalidInputException(NLPModelException):
    """Exception for invalid input"""
    pass

class ModelNotTrainedException(NLPModelException):
    """Exception for model not trained"""
    pass

class NLPModel:
    """
    Main NLP model implementation.

    Attributes:
    ----------
    model : object
        The underlying machine learning model.
    params : dict
        Model parameters.
    """

    def __init__(self, params: Dict):
        """
        Initialize the NLP model.

        Parameters:
        ----------
        params : dict
            Model parameters.
        """
        self.model = None
        self.params = params
        self.trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the NLP model.

        Parameters:
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training labels.
        """
        try:
            # Initialize the model
            self.model = self._initialize_model()

            # Train the model
            self.model.fit(X_train, y_train)

            # Set the trained flag to True
            self.trained = True

            logger.info("Model trained successfully")

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise ModelNotTrainedException("Model not trained")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained NLP model.

        Parameters:
        ----------
        X_test : np.ndarray
            Testing features.

        Returns:
        -------
        np.ndarray
            Predicted labels.
        """
        try:
            if not self.trained:
                raise ModelNotTrainedException("Model not trained")

            # Make predictions
            predictions = self.model.predict(X_test)

            logger.info("Predictions made successfully")

            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the performance of the trained NLP model.

        Parameters:
        ----------
        X_test : np.ndarray
            Testing features.
        y_test : np.ndarray
            Testing labels.

        Returns:
        -------
        Dict
            Evaluation metrics.
        """
        try:
            if not self.trained:
                raise ModelNotTrainedException("Model not trained")

            # Make predictions
            predictions = self.predict(X_test)

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)
            matrix = confusion_matrix(y_test, predictions)

            logger.info("Evaluation metrics calculated successfully")

            return {
                "accuracy": accuracy,
                "report": report,
                "matrix": matrix
            }

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise

    def _initialize_model(self):
        """
        Initialize the underlying machine learning model.

        Returns:
        -------
        object
            The initialized model.
        """
        try:
            # Initialize the model based on the provided parameters
            if self.params["model"] == "logistic_regression":
                model = LogisticRegression(max_iter=1000)
            elif self.params["model"] == "random_forest":
                model = RandomForestClassifier(n_estimators=100)
            elif self.params["model"] == "svm":
                model = SVC(kernel="linear")
            else:
                raise InvalidInputException("Invalid model specified")

            logger.info("Model initialized successfully")

            return model

        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

class NLPDataLoader:
    """
    Data loader for NLP tasks.

    Attributes:
    ----------
    data : pd.DataFrame
        The data to be loaded.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the data loader.

        Parameters:
        ----------
        data : pd.DataFrame
            The data to be loaded.
        """
        self.data = data

    def load_data(self) -> Tuple:
        """
        Load the data.

        Returns:
        -------
        Tuple
            The loaded data.
        """
        try:
            # Split the data into features and labels
            X = self.data.drop("label", axis=1)
            y = self.data["label"]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            logger.info("Data loaded successfully")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

class NLPDataPreprocessor:
    """
    Data preprocessor for NLP tasks.

    Attributes:
    ----------
    data : pd.DataFrame
        The data to be preprocessed.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the data preprocessor.

        Parameters:
        ----------
        data : pd.DataFrame
            The data to be preprocessed.
        """
        self.data = data

    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the data.

        Returns:
        -------
        pd.DataFrame
            The preprocessed data.
        """
        try:
            # Preprocess the data
            self.data = self._handle_missing_values()
            self.data = self._encode_categorical_variables()

            logger.info("Data preprocessed successfully")

            return self.data

        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise

    def _handle_missing_values(self) -> pd.DataFrame:
        """
        Handle missing values in the data.

        Returns:
        -------
        pd.DataFrame
            The data with missing values handled.
        """
        try:
            # Handle missing values
            numerical_columns = self.data.select_dtypes(include=["int64", "float64"]).columns
            categorical_columns = self.data.select_dtypes(include=["object"]).columns

            numerical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_transformer, numerical_columns),
                    ("cat", categorical_transformer, categorical_columns)
                ]
            )

            self.data = pd.DataFrame(preprocessor.fit_transform(self.data))

            logger.info("Missing values handled successfully")

            return self.data

        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise

    def _encode_categorical_variables(self) -> pd.DataFrame:
        """
        Encode categorical variables in the data.

        Returns:
        -------
        pd.DataFrame
            The data with categorical variables encoded.
        """
        try:
            # Encode categorical variables
            categorical_columns = self.data.select_dtypes(include=["object"]).columns

            for column in categorical_columns:
                self.data[column] = pd.Categorical(self.data[column]).codes

            logger.info("Categorical variables encoded successfully")

            return self.data

        except Exception as e:
            logger.error(f"Error encoding categorical variables: {e}")
            raise

def main():
    # Load the data
    data = pd.read_csv("data.csv")

    # Preprocess the data
    preprocessor = NLPDataPreprocessor(data)
    data = preprocessor.preprocess_data()

    # Load the data
    data_loader = NLPDataLoader(data)
    X_train, X_test, y_train, y_test = data_loader.load_data()

    # Train the model
    model = NLPModel({"model": "logistic_regression"})
    model.train(X_train, y_train)

    # Evaluate the model
    evaluation_metrics = model.evaluate(X_test, y_test)
    print(evaluation_metrics)

if __name__ == "__main__":
    main()