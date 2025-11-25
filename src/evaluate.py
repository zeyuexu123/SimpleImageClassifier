import torch
from data import get_test_loader
from model import Net
from config import DEVICE, MODEL_PATH, CLASSES

def evaluate():
    net = Net().to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net.eval()

    testloader = get_test_loader()

    correct = 0
    total = 0
    correct_per_class = {c: 0 for c in CLASSES}
    total_per_class = {c: 0 for c in CLASSES}

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = net(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for label, pred in zip(labels, preds):
                class_name = CLASSES[label]
                total_per_class[class_name] += 1
                if label == pred:
                    correct_per_class[class_name] += 1

    print(f"Overall Accuracy: {100 * correct / total:.2f}%")

    for c in CLASSES:
        acc = 100 * correct_per_class[c] / total_per_class[c]
        print(f"{c}: {acc:.1f}%")