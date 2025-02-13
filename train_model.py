import pandas as pd
import torch
from torch import nn, optim
from ai_detector import AIVulnerabilityDetector
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load data
data = pd.read_csv("data/training_data.csv")
texts = data["text"].tolist()
labels = data["label"].tolist()

# Vectorize text
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts).toarray()

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

# Train model
model = AIVulnerabilityDetector()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# Save model
model.save_model("models/shadow_sense_ai")
joblib.dump(vectorizer, "models/shadow_sense_ai_vectorizer.pkl")
print("Model trained and saved!")
