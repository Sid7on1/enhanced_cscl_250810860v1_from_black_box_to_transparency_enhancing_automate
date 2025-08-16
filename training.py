import logging
import os
import sys
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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import has_fit_parameter, check_scoring
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """
    Training pipeline class for NLP project.

    Attributes:
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained.
    device : torch.device
        The device to be used for training (e.g., CPU or GPU).
    optimizer : torch.optim.Optimizer
        The optimizer to be used for training.
    loss_fn : torch.nn.Module
        The loss function to be used for training.
    train_loader : torch.utils.data.DataLoader
        The data loader for the training dataset.
    val_loader : torch.utils.data.DataLoader
        The data loader for the validation dataset.
    test_loader : torch.utils.data.DataLoader
        The data loader for the testing dataset.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module):
        """
        Initializes the training pipeline.

        Parameters:
        ----------
        model : torch.nn.Module
            The PyTorch model to be trained.
        device : torch.device
            The device to be used for training (e.g., CPU or GPU).
        optimizer : torch.optim.Optimizer
            The optimizer to be used for training.
        loss_fn : torch.nn.Module
            The loss function to be used for training.
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def load_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame, batch_size: int):
        """
        Loads the training, validation, and testing data.

        Parameters:
        ----------
        train_data : pd.DataFrame
            The training dataset.
        val_data : pd.DataFrame
            The validation dataset.
        test_data : pd.DataFrame
            The testing dataset.
        batch_size : int
            The batch size to be used for training.
        """
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    def train(self, epochs: int):
        """
        Trains the model.

        Parameters:
        ----------
        epochs : int
            The number of epochs to train the model.
        """
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            logger.info(f'Epoch {epoch+1}, Loss: {total_loss / len(self.train_loader)}')
            self.evaluate()

    def evaluate(self):
        """
        Evaluates the model on the validation dataset.
        """
        self.model.eval()
        total_correct = 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                _, predicted = torch.max(outputs.scores, dim=1)
                total_correct += (predicted == labels).sum().item()
        accuracy = total_correct / len(self.val_loader.dataset)
        logger.info(f'Validation Accuracy: {accuracy:.4f}')

    def test(self):
        """
        Evaluates the model on the testing dataset.
        """
        self.model.eval()
        total_correct = 0
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                _, predicted = torch.max(outputs.scores, dim=1)
                total_correct += (predicted == labels).sum().item()
        accuracy = total_correct / len(self.test_loader.dataset)
        logger.info(f'Testing Accuracy: {accuracy:.4f}')

class NLPDataset(Dataset):
    """
    Custom dataset class for NLP project.

    Attributes:
    ----------
    data : pd.DataFrame
        The dataset.
    tokenizer : transformers.AutoTokenizer
        The tokenizer to be used for preprocessing.
    """

    def __init__(self, data: pd.DataFrame, tokenizer: transformers.AutoTokenizer):
        """
        Initializes the dataset.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset.
        tokenizer : transformers.AutoTokenizer
            The tokenizer to be used for preprocessing.
        """
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class NLPModel(nn.Module):
    """
    Custom model class for NLP project.

    Attributes:
    ----------
    transformer : transformers.AutoModel
        The transformer model to be used as the backbone.
    classifier : nn.Module
        The classifier to be used for prediction.
    """

    def __init__(self, transformer: transformers.AutoModel, classifier: nn.Module):
        """
        Initializes the model.

        Parameters:
        ----------
        transformer : transformers.AutoModel
            The transformer model to be used as the backbone.
        classifier : nn.Module
            The classifier to be used for prediction.
        """
        super(NLPModel, self).__init__()
        self.transformer = transformer
        self.classifier = classifier

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        outputs = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(outputs, labels)
        return outputs, loss

def main():
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('val.csv')
    test_data = pd.read_csv('test.csv')

    # Create the dataset and data loader
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = NLPDataset(train_data, tokenizer)
    val_dataset = NLPDataset(val_data, tokenizer)
    test_dataset = NLPDataset(test_data, tokenizer)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create the model and optimizer
    transformer = transformers.AutoModel.from_pretrained('bert-base-uncased')
    classifier = nn.Linear(transformer.config.hidden_size, 8)
    model = NLPModel(transformer, classifier)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Create the training pipeline
    pipeline = TrainingPipeline(model, device, optimizer, nn.CrossEntropyLoss())
    pipeline.load_data(train_data, val_data, test_data, batch_size)

    # Train the model
    epochs = 5
    pipeline.train(epochs)

    # Evaluate the model
    pipeline.test()

if __name__ == '__main__':
    main()