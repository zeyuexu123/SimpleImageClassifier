import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import BATCH_SIZE, DATA_ROOT, NUM_WORKERS

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_train_loader():
    trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True,
                                            download=True, transform=transform)
    return DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

def get_test_loader():
    testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False,
                                            download=True, transform=transform)
    return DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)