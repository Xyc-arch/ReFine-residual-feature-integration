import os
from torchvision.datasets import USPS, MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

# Root directory where datasets will be stored
data_root = "./data"

# Make sure the directory exists
os.makedirs(data_root, exist_ok=True)

# Common transform: convert PIL images to tensors, normalize if you like
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))  # uncomment to normalize to [-1,1]
])

# Download USPS
usps_train = USPS(root=data_root,
                  train=True,
                  download=True,
                  transform=transform)
usps_test = USPS(root=data_root,
                 train=False,
                 download=True,
                 transform=transform)

# Download MNIST
mnist_train = MNIST(root=data_root,
                    train=True,
                    download=True,
                    transform=transform)
mnist_test = MNIST(root=data_root,
                   train=False,
                   download=True,
                   transform=transform)

# (Optional) Wrap in DataLoaders
batch_size = 64
usps_train_loader = DataLoader(usps_train, batch_size=batch_size, shuffle=True)
usps_test_loader  = DataLoader(usps_test,  batch_size=batch_size, shuffle=False)
mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
mnist_test_loader  = DataLoader(mnist_test,  batch_size=batch_size, shuffle=False)

print("USPS train / test sizes:", len(usps_train), len(usps_test))
print("MNIST train / test sizes:", len(mnist_train), len(mnist_test))
