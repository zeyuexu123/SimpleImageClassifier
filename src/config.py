import torch

BATCH_SIZE = 16
LR = 0.01
MOMENTUM = 0.9
EPOCHS = 8
NUM_WORKERS = 0
DATA_ROOT = './data'
MODEL_PATH = './cifar_net.pth'

DEVICE = torch.device('cuda:0')

CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')