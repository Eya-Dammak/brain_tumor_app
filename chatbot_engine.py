# chatbot_engine.py

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import nltk

nltk.download('punkt')
nltk.download('wordnet')


class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.function_mappings = function_mappings
        self.X = None
        self.y = None

    def tokenize_and_lemmatize(self, text):
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import wordpunct_tokenize

        lemmatizer = WordNetLemmatizer()
        words = wordpunct_tokenize(text)
        return [lemmatizer.lemmatize(word.lower()) for word in words]

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        with open(self.intents_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for intent in data['intents']:
            tag = intent['tag']
            if tag not in self.intents:
                self.intents.append(tag)
                self.intents_responses[tag] = intent['responses']

            for pattern in intent['patterns']:
                words = self.tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(words)
                self.documents.append((words, tag))

        self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        labels = []

        for words, tag in self.documents:
            bag = self.bag_of_words(words)
            bags.append(bag)
            labels.append(self.intents.index(tag))

        self.X = np.array(bags)
        self.y = np.array(labels)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({
                'input_size': self.X.shape[1],
                'output_size': len(self.intents),
                'vocabulary': self.vocabulary,
                'intents': self.intents
            }, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dims = json.load(f)

        self.vocabulary = dims['vocabulary']
        self.intents = dims['intents']

        self.model = ChatbotModel(dims['input_size'], dims['output_size'])
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Load responses
        with open(self.intents_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for intent in data['intents']:
            self.intents_responses[intent['tag']] = intent['responses']

    def process_message(self, message):
        words = self.tokenize_and_lemmatize(message)
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        with torch.no_grad():
            output = self.model(bag_tensor)
            intent_idx = torch.argmax(output).item()
            intent = self.intents[intent_idx]

        if self.function_mappings and intent in self.function_mappings:
            self.function_mappings[intent]()

        return random.choice(self.intents_responses[intent])
