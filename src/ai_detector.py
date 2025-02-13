import joblib
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn

class AIVulnerabilityDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.vectorizer = TfidfVectorizer(max_features=1000)

    def forward(self, x):
        return self.model(x)

    def predict(self, text):
        vec = self.vectorizer.transform([text]).toarray()[0].astype(np.float32)
        with torch.no_grad():
            return self.model(torch.tensor(vec)).item()

    def save_model(self, path):
        torch.save(self.state_dict(), f"{path}_model.pth")
        joblib.dump(self.vectorizer, f"{path}_vectorizer.pkl")

    @classmethod
    def load_model(cls, path):
        model = cls()
        model.load_state_dict(torch.load(f"{path}_model.pth"))
        model.vectorizer = joblib.load(f"{path}_vectorizer.pkl")
        return model
