import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import logging
import joblib

class VulnerabilityDataset(Dataset):
    """PyTorch Dataset for vulnerability detection."""
    
    def __init__(self, texts: List[str], labels: List[int], vectorizer):
        self.texts = texts
        self.labels = labels
        self.vectorizer = vectorizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        vector = self.vectorizer.transform([text]).toarray()[0].astype(np.float32)
        return torch.tensor(vector), torch.tensor(label)

class AIVulnerabilityDetector(nn.Module):
    """Neural network-based vulnerability detector."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.vectorizer = None  # Will be set during training
        self.logger = logging.getLogger(__name__)
    
    def train_model(
        self, 
        X: List[str], 
        y: List[int], 
        epochs: int = 10, 
        batch_size: int = 32
    ) -> None:
        """Train the AI model with data."""
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=1000)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Create datasets
        train_dataset = VulnerabilityDataset(X_train, y_train, self.vectorizer)
        test_dataset = VulnerabilityDataset(X_test, y_test, self.vectorizer)
        
        # Train
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        for epoch in range(epochs):
            for vectors, labels in DataLoader(train_dataset, batch_size=batch_size):
                optimizer.zero_grad()
                outputs = self.model(vectors)
                loss = criterion(outputs, labels.float().unsqueeze(1))
                loss.backward()
                optimizer.step()
            
            self.logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
        
        # Evaluate
        with torch.no_grad():
            test_accuracy = sum(
                (self.model(vectors).round() == labels.unsqueeze(1)).sum().item()
                for vectors, labels in DataLoader(test_dataset)
            ) / len(test_dataset)
        
        self.logger.info(f"Test Accuracy: {test_accuracy:.2%}")
    
    def predict(self, text: str) -> float:
        """Predict the probability of a vulnerability."""
        if not self.vectorizer:
            raise ValueError("Model not trained yet!")
        vector = self.vectorizer.transform([text]).toarray()[0].astype(np.float32)
        with torch.no_grad():
            return self.model(torch.tensor(vector)).item()
    
    def save_model(self, path: str) -> None:
        """Save the model and vectorizer."""
        torch.save(self.state_dict(), f"{path}_model.pth")
        joblib.dump(self.vectorizer, f"{path}_vectorizer.pkl")
    
    @classmethod
    def load_model(cls, path: str, input_dim: int = 1000) -> "AIVulnerabilityDetector":
        """Load a pre-trained model."""
        model = cls(input_dim)
        model.load_state_dict(torch.load(f"{path}_model.pth"))
        model.vectorizer = joblib.load(f"{path}_vectorizer.pkl")
        return model
