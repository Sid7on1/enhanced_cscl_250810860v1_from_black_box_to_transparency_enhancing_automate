import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from enhanced_ai.models.base import BaseModel
from enhanced_ai.utils.data import TextDataset
from enhanced_ai.utils.logging import setup_logging
from enhanced_ai.utils.misc import ConfigError, load_config

logger = logging.getLogger(__name__)


def setup_device(device: str) -> torch.device:
    """
    Sets up the device for model training/inference.

    Args:
        device (str): Device to use for computations (cpu or cuda).

    Returns:
        torch.device: Device object representing the chosen device.

    Raises:
        ConfigError: If the specified device is not available.
    """
    device = device.lower()
    if device == "cuda" and torch.cuda.is_available():
        logger.info("Using CUDA device for computations.")
        return torch.device("cuda")
    elif device == "cpu":
        logger.info("Using CPU for computations.")
        return torch.device("cpu")
    else:
        raise ConfigError(f"Invalid or unavailable device: {device}")


class ModelConfig:
    """
    Configuration class for the AI model.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of word embeddings.
        hidden_dim (int): Dimension of hidden layer.
        output_dim (int): Dimension of output layer.
        dropout (float): Dropout probability.
        device (str): Device to use for computations (cpu or cuda).
        num_layers (int): Number of layers in the model.
        bidirectional (bool): Whether to use bidirectional RNNs.
        rnn_type (str): Type of RNN to use (gru or lstm).

    Attributes:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of word embeddings.
        hidden_dim (int): Dimension of hidden layer.
        output_dim (int): Dimension of output layer.
        dropout (float): Dropout probability.
        device (torch.device): Device object representing the chosen device.
        num_layers (int): Number of layers in the model.
        bidirectional (bool): Whether to use bidirectional RNNs.
        rnn_type (str): Type of RNN used in the model.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        device: str,
        num_layers: int,
        bidirectional: bool,
        rnn_type: str,
    ) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.device = setup_device(device)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        # Validate RNN type
        if self.rnn_type not in ["gru", "lstm"]:
            raise ConfigError("Invalid RNN type. Choose either 'gru' or 'lstm'.")

        logger.info("Model configuration loaded.")

    def __repr__(self) -> str:
        config_dict = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "device": self.device,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "rnn_type": self.rnn_type,
        }
        return str(config_dict)


class AIModel(BaseModel):
    """
    AI model for enhanced NLP tasks.

    Args:
        config (ModelConfig): Configuration object for the model.
        dataset (TextDataset): Dataset used for training/evaluation.
        learning_rate (float): Learning rate for optimizer.

    Attributes:
        config (ModelConfig): Model configuration.
        dataset (TextDataset): Dataset used for training/evaluation.
        learning_rate (float): Learning rate for optimizer.
        model (nn.Module): Neural network model.
        optimizer (torch.optim): Optimizer for updating model weights.
        criterion (nn.Module): Loss function for training.
        device (torch.device): Device used for computations.
    """

    def __init__(
        self,
        config: ModelConfig,
        dataset: TextDataset,
        learning_rate: float,
    ) -> None:
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.learning_rate = learning_rate

        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = config.device

        logger.info("AI model initialized.")

    def _build_model(self) -> nn.Module:
        """
        Builds and returns the AI model based on the configuration.

        Returns:
            nn.Module: Initialized AI model.
        """
        # Choose RNN type
        rnn_cell = nn.GRU if self.config.rnn_type == "gru" else nn.LSTM

        # Define the layers of the model
        self.embedding = nn.Embedding(
            self.config.vocab_size, self.config.embedding_dim
        )
        self.rnn = rnn_cell(
            self.config.embedding_dim,
            self.config.hidden_dim,
            num_layers=self.config.num_layers,
            bidirectional=self.config.bidirectional,
            batch_first=True,
        )
        self.fc = nn.Linear(
            self.config.hidden_dim * 2, self.config.output_dim
        )  # *2 for bidirectional

        logger.debug("Model architecture built.")
        return nn.Sequential(self.embedding, self.rnn, self.fc)

    def forward(self, text: List[List[str]]) -> np.array:
        """
        Performs forward pass through the model.

        Args:
            text (List[List[str]]): List of tokenized texts.

        Returns:
            np.array: Model predictions.
        """
        # Convert text to input tensors
        inputs = self._text_to_tensor(text)

        # Perform forward pass
        outputs = self.model(inputs)

        # Apply sigmoid activation to get probabilities
        predictions = torch.sigmoid(outputs).squeeze(1).cpu().numpy()

        return predictions

    def train(self, epochs: int, batch_size: int) -> List[float]:
        """
        Trains the AI model for a specified number of epochs.

        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.

        Returns:
            List[float]: List of training losses for each epoch.
        """
        self.model.train()
        losses = []

        for epoch in range(epochs):
            total_loss = 0
            train_loader = DataLoader(
                self.dataset, batch_size=batch_size, shuffle=True
            )

            for batch in train_loader:
                self.optimizer.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.float().to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            epoch_loss = total_loss / len(train_loader)
            losses.append(epoch_loss)
            logger.info(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

        logger.info("Training completed.")
        return losses

    def evaluate(self, text: List[List[str]]) -> np.array:
        """
        Evaluates the AI model on a given list of texts.

        Args:
            text (List[List[str]]): List of tokenized texts for evaluation.

        Returns:
            np.array: Model predictions for the input texts.
        """
        self.model.eval()
        predictions = self.forward(text)
        return predictions

    def save_model(self, model_path: str) -> None:
        """
        Saves the trained model to a file.

        Args:
            model_path (str): Path to save the model.
        """
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to: {model_path}")

    def load_model(self, model_path: str) -> None:
        """
        Loads a trained model from a file.

        Args:
            model_path (str): Path to the saved model.
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        logger.info(f"Model loaded from: {model_path}")

    def _text_to_tensor(self, text: List[List[str]]) -> torch.Tensor:
        """
        Converts a list of tokenized texts to input tensors.

        Args:
            text (List[List[str]]): List of tokenized texts.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, seq_len) containing token indices.
        """
        token_ids = [self.dataset.text_to_ids(txt) for txt in text]
        tokens = torch.LongTensor(token_ids).to(self.device)
        return tokens


def load_model_config(config_path: str) -> ModelConfig:
    """
    Loads the model configuration from a file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        ModelConfig: Model configuration object.

    Raises:
        ConfigError: If the configuration file is missing or incomplete.
    """
    # Load base configuration
    config = load_config(config_path)

    # Validate and extract model-specific configuration
    try:
        model_config = config["model"]
        vocab_size = model_config["vocab_size"]
        embedding_dim = model_config["embedding_dim"]
        hidden_dim = model_config["hidden_dim"]
        output_dim = model_config.get("output_dim", 1)
        dropout = model_config.get("dropout", 0.5)
        device = model_config.get("device", "cpu")
        num_layers = model_config.get("num_layers", 1)
        bidirectional = model_config.get("bidirectional", False)
        rnn_type = model_config.get("rnn_type", "gru")

        # Validate required fields
        if not all(
            isinstance(val, int) and val > 0
            for val in [vocab_size, embedding_dim, hidden_dim, output_dim]
        ):
            raise ConfigError("Invalid or missing values in model configuration.")

        return ModelConfig(
            vocab_size, embedding_dim, hidden_dim, output_dim, dropout, device, num_layers, bidirectional, rnn_type
        )

    except KeyError as e:
        raise ConfigError(f"Missing configuration field: {e}")


def setup_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Sets up the configuration for the project.

    Args:
        config_path (str, optional): Path to the configuration file. Defaults to None.

    Returns:
        Dict[str, Any]: Project configuration dictionary.

    Raises:
        ConfigError: If the configuration file is missing or incomplete.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    if not os.path.exists(config_path):
        raise ConfigError(f"Configuration file not found: {config_path}")

    # Load and validate configuration
    config = load_config(config_path)

    # Set up logging based on config
    logging_config = config.get("logging", {})
    setup_logging(**logging_config)

    return config


# Example usage
if __name__ == "__main__":
    config_path = "path/to/config.yaml"
    model_config_path = "path/to/model_config.yaml"

    # Load project configuration
    config = setup_config(config_path)

    # Load model configuration
    model_config = load_model_config(model_config_path)

    # Prepare dataset
    dataset = TextDataset(config["dataset"])

    # Create AI model
    ai_model = AIModel(model_config, dataset, learning_rate=0.001)

    # Train the model
    ai_model.train(epochs=10, batch_size=32)

    # Evaluate the model
    texts = [["hello", "world"], ["how", "are", "you"]]
    predictions = ai_model.evaluate(texts)
    print(predictions)  # Example output: [0.7842, 0.6534]

    # Save the trained model
    ai_model.save_model("trained_model.pth")