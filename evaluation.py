import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NLPEvaluationMetrics:
    """
    Class for NLP evaluation metrics.

    Attributes:
    - model (torch.nn.Module): The NLP model to evaluate.
    - tokenizer (transformers.AutoTokenizer): The tokenizer for the model.
    - device (torch.device): The device to use for evaluation.
    """

    def __init__(self, model: torch.nn.Module, tokenizer: AutoTokenizer, device: torch.device):
        """
        Initialize the NLPEvaluationMetrics class.

        Args:
        - model (torch.nn.Module): The NLP model to evaluate.
        - tokenizer (transformers.AutoTokenizer): The tokenizer for the model.
        - device (torch.device): The device to use for evaluation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(self, dataset: Dataset, batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate the model on the given dataset.

        Args:
        - dataset (Dataset): The dataset to evaluate on.
        - batch_size (int): The batch size to use for evaluation. Defaults to 32.

        Returns:
        - Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        try:
            # Create a data loader for the dataset
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # Initialize the evaluation metrics
            total_loss = 0
            total_correct = 0
            total_samples = 0

            # Evaluate the model on the dataset
            with torch.no_grad():
                for batch in data_loader:
                    # Move the batch to the device
                    batch = {key: value.to(self.device) for key, value in batch.items()}

                    # Get the model outputs
                    outputs = self.model(**batch)

                    # Calculate the loss
                    loss = outputs.loss

                    # Calculate the accuracy
                    _, predicted = torch.max(outputs.logits, dim=1)
                    correct = (predicted == batch['labels']).sum().item()

                    # Update the evaluation metrics
                    total_loss += loss.item()
                    total_correct += correct
                    total_samples += batch['labels'].size(0)

            # Calculate the average loss and accuracy
            average_loss = total_loss / len(data_loader)
            accuracy = total_correct / total_samples

            # Calculate the precision, recall, and F1 score
            precision = precision_score(dataset.labels, [1 if label == 1 else 0 for label in dataset.labels])
            recall = recall_score(dataset.labels, [1 if label == 1 else 0 for label in dataset.labels])
            f1 = f1_score(dataset.labels, [1 if label == 1 else 0 for label in dataset.labels])

            # Return the evaluation metrics
            return {
                'loss': average_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        except Exception as e:
            logging.error(f'Error evaluating model: {e}')
            return None

    def calculate_velocity_threshold(self, dataset: Dataset, batch_size: int = 32) -> float:
        """
        Calculate the velocity threshold for the given dataset.

        Args:
        - dataset (Dataset): The dataset to calculate the velocity threshold for.
        - batch_size (int): The batch size to use for calculation. Defaults to 32.

        Returns:
        - float: The velocity threshold.
        """
        try:
            # Create a data loader for the dataset
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # Initialize the velocity threshold
            velocity_threshold = 0

            # Calculate the velocity threshold
            with torch.no_grad():
                for batch in data_loader:
                    # Move the batch to the device
                    batch = {key: value.to(self.device) for key, value in batch.items()}

                    # Get the model outputs
                    outputs = self.model(**batch)

                    # Calculate the velocity
                    velocity = torch.mean(outputs.logits)

                    # Update the velocity threshold
                    velocity_threshold += velocity.item()

            # Calculate the average velocity threshold
            velocity_threshold /= len(data_loader)

            # Return the velocity threshold
            return velocity_threshold

        except Exception as e:
            logging.error(f'Error calculating velocity threshold: {e}')
            return None

    def calculate_flow_theory(self, dataset: Dataset, batch_size: int = 32) -> float:
        """
        Calculate the flow theory for the given dataset.

        Args:
        - dataset (Dataset): The dataset to calculate the flow theory for.
        - batch_size (int): The batch size to use for calculation. Defaults to 32.

        Returns:
        - float: The flow theory.
        """
        try:
            # Create a data loader for the dataset
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # Initialize the flow theory
            flow_theory = 0

            # Calculate the flow theory
            with torch.no_grad():
                for batch in data_loader:
                    # Move the batch to the device
                    batch = {key: value.to(self.device) for key, value in batch.items()}

                    # Get the model outputs
                    outputs = self.model(**batch)

                    # Calculate the flow
                    flow = torch.mean(outputs.logits)

                    # Update the flow theory
                    flow_theory += flow.item()

            # Calculate the average flow theory
            flow_theory /= len(data_loader)

            # Return the flow theory
            return flow_theory

        except Exception as e:
            logging.error(f'Error calculating flow theory: {e}')
            return None


class NLPEvaluationDataset(Dataset):
    """
    Class for NLP evaluation dataset.

    Attributes:
    - data (List[str]): The list of text data.
    - labels (List[int]): The list of labels.
    - tokenizer (transformers.AutoTokenizer): The tokenizer for the model.
    """

    def __init__(self, data: List[str], labels: List[int], tokenizer: AutoTokenizer):
        """
        Initialize the NLPEvaluationDataset class.

        Args:
        - data (List[str]): The list of text data.
        - labels (List[int]): The list of labels.
        - tokenizer (transformers.AutoTokenizer): The tokenizer for the model.
        """
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
        - int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get the item at the given index.

        Args:
        - index (int): The index of the item.

        Returns:
        - Dict[str, torch.Tensor]: The item at the given index.
        """
        try:
            # Get the text and label at the given index
            text = self.data[index]
            label = self.labels[index]

            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)

            # Create a dictionary for the item
            item = {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }

            # Return the item
            return item

        except Exception as e:
            logging.error(f'Error getting item: {e}')
            return None


def main():
    # Load the dataset
    data = pd.read_csv('data.csv')

    # Split the dataset into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

    # Create a tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Create a model
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Create a device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model.to(device)

    # Create a dataset for the training data
    train_dataset = NLPEvaluationDataset(train_data, train_labels, tokenizer)

    # Create a dataset for the testing data
    test_dataset = NLPEvaluationDataset(test_data, test_labels, tokenizer)

    # Create an evaluation metrics class
    evaluation_metrics = NLPEvaluationMetrics(model, tokenizer, device)

    # Evaluate the model on the testing data
    evaluation_results = evaluation_metrics.evaluate(test_dataset)

    # Print the evaluation results
    print(evaluation_results)

    # Calculate the velocity threshold
    velocity_threshold = evaluation_metrics.calculate_velocity_threshold(test_dataset)

    # Print the velocity threshold
    print(f'Velocity Threshold: {velocity_threshold}')

    # Calculate the flow theory
    flow_theory = evaluation_metrics.calculate_flow_theory(test_dataset)

    # Print the flow theory
    print(f'Flow Theory: {flow_theory}')


if __name__ == '__main__':
    main()