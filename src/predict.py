
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import DigitCNN


def predict_digit(image_path):
    model = DigitCNN()
    model.load_state_dict(
        torch.load("mnist_model.pth", map_location="cpu", weights_only=True)
    )
    model.eval()

    original = Image.open(image_path).convert("L")
    arr = np.array(original)

    threshold = arr.mean()
    binary = np.where(arr < threshold, 255, 0).astype(np.uint8)

    coords = np.argwhere(binary > 0)

    if len(coords) == 0:
        print("Digit could not be detected.")
        return

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    digit = binary[y_min:y_max + 1, x_min:x_max + 1]
    digit_img = Image.fromarray(digit)
    digit_img.thumbnail((18, 18))

    mnist_img = Image.new("L", (28, 28), 0)
    x = (28 - digit_img.width) // 2
    y = (28 - digit_img.height) // 2
    mnist_img.paste(digit_img, (x, y))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    input_image = transform(mnist_img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    print("Predicted Digit:", predicted.item())
    print(f"Confidence: {confidence.item() * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py image.jpg")
    else:
        predict_digit(sys.argv[1])
