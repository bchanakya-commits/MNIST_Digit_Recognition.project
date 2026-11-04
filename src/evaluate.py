
import torch

from model import DigitCNN
from data_preprocessing import load_data


def evaluate_model():
    _, test_loader = load_data()

    model = DigitCNN()
    model.load_state_dict(
        torch.load("mnist_model.pth", map_location="cpu", weights_only=True)
    )

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    evaluate_model()
