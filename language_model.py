import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
import pandas as pd
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
class Config(Enum):
    MODEL_NAME = 'bert-base-uncased'
    MAX_SEQ_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    CLASS_WEIGHTS = None

class LanguageModel:
    def __init__(self, config: Config):
        self.config = config
        self.model_name = config.MODEL_NAME
        self.max_seq_length = config.MAX_SEQ_LENGTH
        self.batch_size = config.BATCH_SIZE
        self.epochs = config.EPOCHS
        self.learning_rate = config.LEARNING_RATE
        self.weight_decay = config.WEIGHT_DECAY
        self.class_weights = config.CLASS_WEIGHTS
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[List[str], List[int]]:
        # Preprocess text data
        texts = data['text'].tolist()
        labels = data['label'].tolist()
        le = LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        return texts, labels

    def create_dataset(self, texts: List[str], labels: List[int]) -> Dataset:
        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_seq_length):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_seq_length = max_seq_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                label = self.labels[idx]

                encoding = self.tokenizer.encode_plus(
                    text,
                    max_length=self.max_seq_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )

                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long),
                }

        return TextDataset(texts, labels, self.tokenizer, self.max_seq_length)

    def train(self, train_data: Dataset, val_data: Dataset) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        for epoch in range(self.epochs):
            logger.info(f'Epoch {epoch+1}/{self.epochs}')
            start_time = time.time()

            train_loss = 0
            self.model.train()
            for batch in DataLoader(train_data, batch_size=self.batch_size, shuffle=True):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_data)
            logger.info(f'Train Loss: {train_loss:.4f}')

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in DataLoader(val_data, batch_size=self.batch_size, shuffle=False):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                    val_loss += loss.item()

            val_loss /= len(val_data)
            logger.info(f'Val Loss: {val_loss:.4f}')

            scheduler.step()

            end_time = time.time()
            logger.info(f'Time taken for epoch {epoch+1}: {end_time - start_time:.2f} seconds')

    def evaluate(self, test_data: Dataset) -> None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        self.model.eval()
        test_loss = 0
        predictions = []
        labels = []
        with torch.no_grad():
            for batch in DataLoader(test_data, batch_size=self.batch_size, shuffle=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_batch = batch['labels'].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels_batch)
                loss = outputs.loss

                test_loss += loss.item()

                logits = outputs.logits
                _, predicted = torch.max(logits, dim=1)
                predictions.extend(predicted.cpu().numpy())
                labels.extend(labels_batch.cpu().numpy())

        test_loss /= len(test_data)
        logger.info(f'Test Loss: {test_loss:.4f}')

        accuracy = accuracy_score(labels, predictions)
        logger.info(f'Test Accuracy: {accuracy:.4f}')

        report = classification_report(labels, predictions)
        logger.info(report)

        matrix = confusion_matrix(labels, predictions)
        logger.info(matrix)

    def save_model(self, path: str) -> None:
        self.model.save_pretrained(path)

    def load_model(self, path: str) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)

def main():
    config = Config()
    model = LanguageModel(config)

    # Load data
    data = pd.read_csv('data.csv')
    texts, labels = model.preprocess_data(data)

    # Split data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Create datasets and data loaders
    train_dataset = model.create_dataset(train_texts, train_labels)
    val_dataset = model.create_dataset(val_texts, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Train the model
    model.train(train_dataset, val_dataset)

    # Evaluate the model
    test_dataset = model.create_dataset(texts, labels)
    model.evaluate(test_dataset)

    # Save the model
    model.save_model('model')

if __name__ == '__main__':
    main()