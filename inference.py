import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from Server.models import Discriminator
from torchvision.datasets import CIFAR10

transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])])
test_dataset = CIFAR10(root='data/cifar10', train=False, download=False, transform=transform)

data_loader = DataLoader(test_dataset, shuffle=True)
print(data_loader)

for client in range(5):
    model = Discriminator()
    path = 'savedModels//Discriminator{}_state_dict_model.pt'.format(str(client))
    model.load_state_dict(torch.load(path))
    model.eval()
    for i, (data, _) in enumerate(data_loader):
        print("D:", str(client), "data:", data, "Result:", model(data))