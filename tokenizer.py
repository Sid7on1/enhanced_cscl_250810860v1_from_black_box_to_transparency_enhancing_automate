# tokenizer.py
"""
Text tokenization utilities.
"""

import logging
import re
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import torch
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenizerType(Enum):
    """Tokenizer type."""
    BERT = "bert-base-uncased"
    ROBERTA = "roberta-base"

class Tokenizer(ABC):
    """Abstract base class for tokenizers."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = self._create_tokenizer()

    @abstractmethod
    def _create_tokenizer(self) -> AutoTokenizer:
        """Create a tokenizer instance."""
        pass

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a text."""
        return self.tokenizer.tokenize(text)

    def encode(self, text: str) -> torch.Tensor:
        """Encode a text."""
        return self.tokenizer.encode(text, return_tensors="pt")

class BertTokenizer(Tokenizer):
    """BERT tokenizer."""
    def _create_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.model_name)

class RobertaTokenizer(Tokenizer):
    """RoBERTa tokenizer."""
    def _create_tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.model_name)

class WordpieceTokenizer:
    """Wordpiece tokenizer."""
    def __init__(self, vocab_file: str, max_input_chars_per_word: int = 100):
        self.vocab_file = vocab_file
        self.max_input_chars_per_word = max_input_chars_per_word
        self.vocab = self._load_vocab()

    def _load_vocab(self) -> Dict[str, int]:
        with open(self.vocab_file, "r") as f:
            vocab = {}
            for line in f:
                token = line.strip().split()[0]
                vocab[token] = len(vocab)
        return vocab

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        for token in re.findall(r"\S+", text):
            if len(token) > self.max_input_chars_per_word:
                sub_tokens = []
                for i in range(0, len(token), self.max_input_chars_per_word):
                    sub_token = token[i:i + self.max_input_chars_per_word]
                    sub_tokens.append(sub_token)
                tokens.extend(sub_tokens)
            else:
                tokens.append(token)
        return tokens

class WordpieceEncoder:
    """Wordpiece encoder."""
    def __init__(self, vocab_file: str):
        self.vocab_file = vocab_file
        self.vocab = self._load_vocab()

    def _load_vocab(self) -> Dict[str, int]:
        with open(self.vocab_file, "r") as f:
            vocab = {}
            for line in f:
                token = line.strip().split()[0]
                vocab[token] = len(vocab)
        return vocab

    def encode(self, text: str) -> torch.Tensor:
        tokens = WordpieceTokenizer(self.vocab_file).tokenize(text)
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab["[UNK]"])
        return torch.tensor(ids)

class TokenizerConfig:
    """Tokenizer configuration."""
    def __init__(self, model_name: str, vocab_file: str):
        self.model_name = model_name
        self.vocab_file = vocab_file

class TokenizerManager:
    """Tokenizer manager."""
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.tokenizers = {}

    def get_tokenizer(self, model_name: str) -> Tokenizer:
        if model_name not in self.tokenizers:
            if model_name == "bert-base-uncased":
                self.tokenizers[model_name] = BertTokenizer(self.config.model_name)
            elif model_name == "roberta-base":
                self.tokenizers[model_name] = RobertaTokenizer(self.config.model_name)
            else:
                raise ValueError(f"Unsupported model name: {model_name}")
        return self.tokenizers[model_name]

def main():
    config = TokenizerConfig(model_name="bert-base-uncased", vocab_file="vocab.txt")
    manager = TokenizerManager(config)
    tokenizer = manager.get_tokenizer("bert-base-uncased")
    text = "This is a sample text."
    tokens = tokenizer.tokenize(text)
    logger.info(f"Tokens: {tokens}")
    encoded = tokenizer.encode(text)
    logger.info(f"Encoded: {encoded}")

if __name__ == "__main__":
    main()