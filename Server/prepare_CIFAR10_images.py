from torchvision import datasets
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image

transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])])
dataset = datasets.CIFAR10(root='data/cifar10', train=True, download=False, transform=transform)

Path("data/imgs").mkdir(parents=False, exist_ok=True)

for idx, (img, _) in enumerate(dataset):
    save_image(img, 'data/imgs/{:05d}.jpg'.format(idx))
  
