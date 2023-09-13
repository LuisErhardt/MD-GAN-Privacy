import torch
import csv
from torch.utils.data import DataLoader
from torchvision import transforms
from Server.models import Discriminator
from torchvision.datasets import CIFAR10
from Server.FID_evaluation_CIFAR10 import array_of_epochs

transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])])
test_dataset = CIFAR10(root='data/cifar10', train=False, download=False, transform=transform)
train_dataset = CIFAR10(root='data/cifar10', train=True, download=False, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

test_batch = next(iter(test_loader))
train_batch = next(iter(train_loader))
print("Test batch:", test_batch)
print("Train batch:", test_batch)

results=[]

for i, (batch, name) in enumerate([(test_batch, "test_batch"), (train_batch, "train_batch")]):
    for epoch in array_of_epochs(20,15):
        for client in range(5):
            path = 'savedModels/epoch{}/Discriminator{}_state_dict_model.pt'.format(epoch, str(client))
            model = Discriminator()
            model.load_state_dict(torch.load(path))
            model.eval()
            data, _ = batch
            score = model(data)
            print("Epoch:", epoch, "D:", str(client), "Batch:", name, "Score:", score.item())
            results.append([epoch, client, name, score.item()])

csv_filename = "result_inference.csv"

# Open the CSV file in write mode
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(["Epoch", "Discriminator", "Batch", "Score"])  # Uncomment this line if you want headers

    for row in results:
        writer.writerow(row)