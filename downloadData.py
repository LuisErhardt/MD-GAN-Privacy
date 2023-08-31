from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
import argparse



def download_Dataset(dataset):

    for directory in ["Server/","BenignClient/"]:

        transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])])
        if dataset == 'cifar100':
            train_dataset = CIFAR100(root=directory + 'data/cifar100', train=True, download=True, transform=transform)
            # test_dataset = CIFAR10(root=directory + 'data/cifar10', train=False, download=False, transform=transform)
        if dataset == 'cifar10':
            train_dataset = CIFAR10(root=directory + 'data/cifar10', train=True, download=True, transform=transform)
            # test_dataset = CIFAR10(root=directory + 'data/cifar10', train=False, download=False, transform=transform)


        if dataset == 'mnist':
            train_dataset = MNIST(root=directory + 'data/mnist', train=True, download=True, transform=transform)
            # test_dataset = MNIST(root=directory + 'data/mnist', train=False, download=False, transform=transform)

        if dataset == 'fashion':
            train_dataset = FashionMNIST(root=directory + 'data/fashion', train=True, download=True, transform=transform)
            # test_dataset = FashionMNIST(root=directory + 'data/fashion', train=False, download=False, transform=transform)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dataset", type=str, default="mnist"
    )
    args = parser.parse_args()

    download_Dataset(args.dataset)