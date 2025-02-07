import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
from ai_detector import AIVulnerabilityDetector

# Load training data
data = pd.read_csv("data/training_data.csv")
texts = data["text"].tolist()
labels = data["label"].tolist()

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_vec.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val_vec.toarray(), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# Define dataset
class VulnerabilityDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = VulnerabilityDataset(X_train_tensor, y_train_tensor)
val_dataset = VulnerabilityDataset(X_val_tensor, y_val_tensor)

# Initialize model
model = AIVulnerabilityDetector(input_dim=1000)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

for epoch in range(10):
    # Train
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            val_loss += criterion(outputs, y_batch).item()
            correct += (outputs.round() == y_batch).sum().item()
    
    # Print metrics
    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"Val Accuracy: {correct/len(val_dataset):.2%}\n")

# Save the trained model and vectorizer
model.save_model("models/shadow_sense_ai")
joblib.dump(vectorizer, "models/shadow_sense_ai_vectorizer.pkl")
print("Model saved to models/ directory!")
