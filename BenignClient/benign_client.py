import argparse
import torch
import pandas as pd
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch.distributed.rpc as rpc
from torch.autograd import Variable
import torch.distributed as dist
from torchvision.utils import make_grid
from torch import autograd
import imageio
from models import Generator, Discriminator
# from wgangp import WGANGP
import numpy as np
import os

import warnings

warnings.filterwarnings("ignore")


def _call_method(method, rref, *args, **kwargs):
    """helper for _remote_method()"""
    return method(rref.local_value(), *args, **kwargs)
def _remote_method(method, rref, *args, **kwargs):
    """
    executes method(*args, **kwargs) on the from the machine that owns rref
    very similar to rref.remote().method(*args, **kwargs), but method() doesn't have to be in the remote scope
    """
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)
def param_rrefs(module):
    """grabs remote references to the parameters of a module"""
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(rpc.RRef(param))
    print(param_rrefs)
    return param_rrefs

            
class MDGANClient():
    """
    This is the class that encapsulates the functions that need to be run on the client side
    Despite the name, this source code only needs to reside on the server and will be executed via RPC.
    """

    def __init__(self, dataset, epochs, use_cuda, batch_size, **kwargs):
        self.epochs = epochs
        self.latent_shape = [100, 1, 1]
        self.use_cuda = False
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type != 'cpu':
             self.use_cuda = True
        transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])])
        if dataset == 'cifar10':
            train_dataset = CIFAR10(root='data/cifar10', train=True, download=False, transform=transform)
            # test_dataset = CIFAR10(root='data/cifar10', train=False, download=False, transform=transform)


        if dataset == 'mnist':
            train_dataset = MNIST(root='data/mnist', train=True, download=False, transform=transform)
            # test_dataset = MNIST(root='data/mnist', train=False, download=False, transform=transform)

        if dataset == 'fashion':
            train_dataset = FashionMNIST(root='data/fashion', train=True, download=False, transform=transform)
            # test_dataset = FashionMNIST(root='data/fashion', train=False, download=False, transform=transform)

        full_dataset = train_dataset
        print(full_dataset.targets[0:10])

        idx_list = []
        final_chosen_list = None
        seed = np.random.choice(1000,1)[0]
        print("seed: ", seed)
        for id_ in range(10):
            ToChoseIndex = np.where(np.array(full_dataset.targets) == id_)[0]
            np.random.seed(seed)
            chosen_index_list = np.random.choice(ToChoseIndex, 500, replace=False)
            print("chosen data: ", id_, np.array(full_dataset.targets)[chosen_index_list[0:10]])
            if final_chosen_list is None:
                final_chosen_list = chosen_index_list
            else:
                final_chosen_list = np.concatenate([final_chosen_list, chosen_index_list])
        print("final chosen list length: ", final_chosen_list)
        full_dataset.data, full_dataset.targets = full_dataset.data[final_chosen_list], list(np.array(full_dataset.targets)[final_chosen_list])
        
        self.steps_per_epoch = len(full_dataset) // batch_size
        print("steps_per_epoch: ", self.steps_per_epoch)
        self.data_loader = DataLoader(full_dataset, batch_size = batch_size, shuffle=True)
        self.discriminator = Discriminator()
        if self.device.type != 'cpu':
            print(self.device.type)
            self.discriminator.cuda()
        self.D_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))


    def send_client_refs(self):
        """Send a reference to the discriminator (for future RPC calls) and the conditioner, transformer and steps/epoch"""
        return rpc.RRef(self.discriminator)


    def register_G(self, G_rref):
        """Receive a reference to the generator (for future RPC calls)"""
        self.G_rref = G_rref
        
    def sample_latent(self):
        z = Variable(torch.randn(self.batch_size, *self.latent_shape))
        return z

    def get_discriminator_weights(self):
        print("call in get_discriminator_weights: ")
        if next(self.discriminator.parameters()).is_cuda:
            return self.discriminator.cpu().state_dict()
        else:
            return self.discriminator.state_dict()
    def set_discriminator_weights(self, state_dict):
        print("call in set_discriminator_weights: ", self.device)
        self.discriminator.load_state_dict(state_dict)
        if self.device.type != 'cpu':
            print("set discriminator on cuda")
            self.discriminator.to(self.device)
    def reset_on_cuda(self):
        if self.device.type != 'cpu':
            self.discriminator.to(self.device)

    def get_steps_number(self):
        return self.steps_per_epoch

    def gradient_penalty(self, data, generated_data, gamma=10):
        batch_size = data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1)
        epsilon = epsilon.expand_as(data)


        if self.use_cuda:
            epsilon = epsilon.cuda()

        interpolation = epsilon * data.data + (1 - epsilon) * generated_data.data
        interpolation = Variable(interpolation, requires_grad=True)

        if self.use_cuda:
            interpolation = interpolation.cuda()
        # print("shape of data", data.shape, generated_data.shape, interpolation.shape)
        interpolation_logits = self.discriminator(interpolation)
        grad_outputs = torch.ones(interpolation_logits.size())

        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = autograd.grad(outputs=interpolation_logits,
                                  inputs=interpolation,
                                  grad_outputs=grad_outputs,
                                  create_graph=True,
                                  retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        # here add an epsilon to avoid sqrt error for number close to 0
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12) 
        return gamma * ((gradients_norm - 1) ** 2).mean()

    def loss_G_data(self, generated_data):
        # generated_data = self.G_rref.remote().forward(self.sample_latent().cpu()).to_here()
        generated_data = generated_data.to(self.device)
        g_loss = -self.discriminator(generated_data).mean()
        return g_loss.cpu(), self.discriminator(generated_data).cpu().tolist()


    def loss_G(self):
        generated_data = self.G_rref.remote().forward(self.sample_latent().cpu()).to_here()
        generated_data = generated_data.to(self.device)
        g_loss = -self.discriminator(generated_data).mean()
        return g_loss.cpu()

    def evalute_D(self, testing_latent):
        if self.device.type != 'cpu':
            testing_latent = testing_latent.cuda()
            # print("discriminator output dimension: ", self.discriminator(testing_latent)[0:10])
            return self.discriminator(testing_latent).cpu().tolist()
        else:
            return self.discriminator(testing_latent).tolist()

    def train_D(self):
        
        iterloader = iter(self.data_loader)
        try:
            data, _ = next(iterloader)
        except StopIteration:
            iterloader = iter(self.data_loader)
            data, _ = next(iterloader)
            
        generated_data = self.G_rref.remote().forward(self.sample_latent().cpu()).to_here()
        generated_data = generated_data.to(device=self.device)
        data = data.to(self.device)

        grad_penalty = self.gradient_penalty(data, generated_data)
        d_loss = self.discriminator(generated_data).mean() - self.discriminator(data).mean() + grad_penalty
        self.D_opt.zero_grad()
        d_loss.backward()
        self.D_opt.step()

        return d_loss.item(), grad_penalty.item()
    


def run(rank, world_size, ip, port, dataset, epochs, use_cuda, batch_size, n_critic):
    # set environment information
    os.environ["MASTER_ADDR"] = ip
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    # print("number of epochs before initialization: ", epochs)
    # print("world size: ", world_size, f"tcp://{ip}:{port}")
    if rank != 0:
        rpc.init_rpc(
            "client"+str(rank),
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                num_worker_threads=8, rpc_timeout=120, init_method=f"tcp://{ip}:{port}", _transports=["uv"]
            ),
        )
        print("Client"+str(rank)+" joined")

    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-rank", type=int, default=1)
    parser.add_argument("-ip", type=str, default="127.0.0.1")
    parser.add_argument("-port", type=int, default=7788)
    parser.add_argument(
        "-dataset", type=str, default="mnist"
    )
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-world_size", type=int, default=2)
    parser.add_argument('-use_cuda',  type=str, default='True')
    parser.add_argument("-batch_size", type=int, default=500)
    parser.add_argument("-n_critic", type=int, default=1)
    args = parser.parse_args()

    if args.rank is not None:
        # run with a specified rank (need to start up another process with the opposite rank elsewhere)
        run(
            rank=args.rank,
            world_size=args.world_size,
            ip=args.ip,
            port=args.port,
            dataset=args.dataset,
            epochs=args.epochs,
            use_cuda=args.use_cuda,
            batch_size=args.batch_size,
            n_critic=args.n_critic

        )
