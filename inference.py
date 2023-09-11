import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from Server.models import Discriminator
from torchvision.datasets import CIFAR10
from Server.FID_evaluation_CIFAR10 import array_of_epochs

transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])])
test_dataset = CIFAR10(root='data/cifar10', train=False, download=False, transform=transform)

data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
batch1 = next(iter(data_loader))
batch2 = next(iter(data_loader))
print("Batch1:", batch1)
print("Batch2:", batch2)

for epoch in array_of_epochs(20,15):
    for client in range(5):
        path = 'savedModels/epoch{}/Discriminator{}_state_dict_model.pt'.format(epoch, str(client))
        model = Discriminator()
        model.load_state_dict(torch.load(path))
        model.eval()
        for i, batch in enumerate([batch1, batch2]):
            data, _ = batch
            score = model(data)
            print("Epoch:", epoch, "D:", str(client), "Batch:", i, "Score:", score)
