
import torch
import torch.nn as nn
import torch.optim as optim

from model import DigitCNN
from data_preprocessing import load_data


def train_model():
    train_loader, _ = load_data()

    model = DigitCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {avg_loss:.4f}"
        )

    torch.save(model.state_dict(), "mnist_model.pth")
    print("Model training completed and saved!")


if __name__ == "__main__":
    train_model()
