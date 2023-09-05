import argparse
import torch
import pandas as pd
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed.rpc as rpc
from torch.autograd import Variable
import torch.distributed as dist
from torch.distributed.optim import DistributedOptimizer
import torch.optim as optim
from torchvision.utils import make_grid
from torch import autograd
import glob
from PIL import Image
from models import Generator, Discriminator
from torchvision.utils import save_image
# from wgangp import WGANGP
from sklearn.cluster import KMeans
import numpy as np
import os
import csv
from scipy.spatial import distance_matrix
from pathlib import Path


import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

# parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion'])
# parser.add_argument('--epochs', type=int, default=200)
# parser.add_argument('--batch_size', type=int, default=64)


# args = parser.parse_args()

# transform = transforms.Compose([
#     transforms.Resize(32),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])

# if args.dataset == 'cifar10':
#     train_dataset = CIFAR10(root='data/cifar10', train=True, download=False, transform=transform)
#     test_dataset = CIFAR10(root='data/cifar10', train=False, download=False, transform=transform)


# if args.dataset == 'mnist':
#     train_dataset = MNIST(root='data/mnist', train=True, download=False, transform=transform)
#     test_dataset = MNIST(root='data/mnist', train=False, download=False, transform=transform)

# if args.dataset == 'fashion':
#     train_dataset = FashionMNIST(root='data/fashion', train=True, download=False, transform=transform)
#     test_dataset = FashionMNIST(root='data/fashion', train=False, download=False, transform=transform)

# full_dataset = ConcatDataset([train_dataset, test_dataset])
# data_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)

# generator = Generator(100)
# discriminator = Discriminator()

# g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
# d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

# wgan = WGANGP(generator, discriminator, g_optimizer, d_optimizer, [100, 1, 1], args.dataset)
# wgan.train(data_loader, args.epochs)

# pd.DataFrame(wgan.hist).to_csv(args.dataset + '_hist.csv', index=False)

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
def sum_of_layer(model_dicts, layer):
    """
    Sum of parameters of one layer for all models
    """
    layer_sum = model_dicts[0][layer]
    for i in range(1, len(model_dicts)):
        layer_sum += model_dicts[i][layer]
    return layer_sum
def average_model(model_dicts):
    """
    Average model by uniform weights
    """
    if len(model_dicts) == 1:
        return model_dicts[0]
    else:
        weights = 1/len(model_dicts)
        state_aggregate = model_dicts[0]
    for layer in state_aggregate:
        state_aggregate[layer] = weights*sum_of_layer(model_dicts, layer)
    return state_aggregate


# def swap_decision_single(user1_id, user2_id, result_vector):
#     '''
#     user1_id: id of user 1
#     user2_id: id of user 2
#     result_vector: distance vector of user1 to all users
#     return: if user1 and user2 are in the same 'benign' group
#     '''
#     kmeans = KMeans(n_clusters=2, random_state=0).fit(result_vector.reshape(-1,1))
#     cluster_map = pd.DataFrame()
#     cluster_map['data'] = result_vector
#     cluster_map['cluster'] = kmeans.labels_
#     cluster_zero = cluster_map[cluster_map.cluster == 0]
#     cluster_one = cluster_map[cluster_map.cluster == 1]
#     if cluster_one['data'].mean() >= cluster_zero['data'].mean():
#         if user2_id in cluster_zero.index:
#             # return True only if user2_id is in the group where their average distance to user1_id is smaller
#             return True
#         else:
#             return False
#     else:
#         if user2_id in cluster_one.index:
#             # return True only if user2_id is in the group where their average distance to user1_id is smaller
#             return True
#         else:
#             return False

def swap_decision_single(user1_id, user2_id, result_vector):
    '''
    user1_id: id of user 1
    user2_id: id of user 2
    result_vector: distance vector of user1 to all users
    return: if user1 and user2 are in the same 'benign' group
    '''
    result_vector = list(result_vector)
    result_vector.remove(0)
    result_vector = np.array(result_vector)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(result_vector.reshape(-1,1))
    labels = [None] * int(result_vector.shape[0]+1)
    result_vector_reconstruction = [None] * int(result_vector.shape[0]+1)
    for i in range(len(result_vector)+1):
        if i < user1_id:
            labels[i] = kmeans.labels_[i]
            result_vector_reconstruction[i] = result_vector[i]
        elif i == user1_id:
            labels[i] = -1
            result_vector_reconstruction[i] = 0
        elif i > user1_id:
            labels[i] = kmeans.labels_[i-1]
            result_vector_reconstruction[i] = result_vector[i-1]   

    cluster_map = pd.DataFrame()
    cluster_map['data'] = result_vector_reconstruction
    cluster_map['cluster'] = labels
    # print("labels: ", labels)
    # print("result_vector_reconstruction: ", result_vector_reconstruction)
    cluster_zero = cluster_map[cluster_map.cluster == 0]
    cluster_one = cluster_map[cluster_map.cluster == 1]
    if cluster_one['data'].mean() >= cluster_zero['data'].mean():
        if user2_id in cluster_zero.index:
            # return True only if user2_id is in the group where their average distance to user1_id is smaller
            return True
        else:
            return False
    else:
        if user2_id in cluster_one.index:
            # return True only if user2_id is in the group where their average distance to user1_id is smaller
            return True
        else:
            return False

def swap_decision(user1_id, user2_id, result_matrix):
    '''
    user1_id: id of user 1
    user2_id: id of user 2
    result_matrix: distance matrix of all users
    return: if allow user1 and user2 to swap their model
    '''

    decision1 = swap_decision_single(user1_id, user2_id, result_matrix[user1_id])
    decision2 = swap_decision_single(user2_id, user1_id, result_matrix[user2_id])
    print("decision1, decison2: ", user1_id, user2_id, decision1, decision2)
    if decision1 and decision2:
        return True
    else:
        return False






class MDGANServer():
    """
    This is the class that encapsulates the functions that need to be run on the server side
    This is the main driver of the training procedure.
    """

    def __init__(self, client_rrefs, epochs, use_cuda, batch_size, n_critic, **kwargs):
        # super(MDGANServer, self).__init__(**kwargs)
        self.epochs = epochs
        self.use_cuda = use_cuda
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.n_critic = n_critic
        self.latent_shape = [100, 1, 1]
        self._fixed_z = torch.randn(64, *self.latent_shape)
        self.batch_size = batch_size
        self.images = []
        self.ignore_clients = []
        self.distance_matrix_records = []
        self.distance_matrix_records_sum = []
        self.ignore_clients_record = []
        self.result_matrix = []
        self.attempted_switch = []
        self.success_switch = []



        dataset = 'cifar10'

        transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])])
        if dataset == 'cifar100':
            train_dataset = CIFAR100(root='data/cifar100', train=True, download=False, transform=transform)
            # test_dataset = CIFAR10(root='data/cifar10', train=False, download=False, transform=transform)
        if dataset == 'cifar10':
            train_dataset = CIFAR10(root='data/cifar10', train=True, download=False, transform=transform)
            # test_dataset = CIFAR10(root='data/cifar10', train=False, download=False, transform=transform)


        if dataset == 'mnist':
            train_dataset = MNIST(root='data/mnist', train=True, download=False, transform=transform)
            test_dataset = MNIST(root='data/mnist', train=False, download=False, transform=transform)

        if dataset == 'fashion':
            train_dataset = FashionMNIST(root='data/fashion', train=True, download=False, transform=transform)
            test_dataset = FashionMNIST(root='data/fashion', train=False, download=False, transform=transform)

        full_dataset = train_dataset
        print(full_dataset.targets[0:10])
        # ConcatDataset([train_dataset, test_dataset])
        idx_list = []
        final_chosen_list = None
        seed = np.random.choice(1000,1)[0]
        # seed = 321
        # seed = 1
        # seed = 853
        print("seed: ", seed)
        for id_ in range(10):
            ToChoseIndex = np.where(np.array(full_dataset.targets) == id_)[0]
            np.random.seed(seed)   
            chosen_index_list = np.random.choice(ToChoseIndex, 500, replace=False)
            if id_ %10 == 0:
                print("chosen data: ", id_, np.array(full_dataset.targets)[chosen_index_list[0:10]])
            if final_chosen_list is None:
                final_chosen_list = chosen_index_list
            else:
                final_chosen_list = np.concatenate([final_chosen_list, chosen_index_list])
        print("final chosen list length: ", final_chosen_list)
        full_dataset.data, full_dataset.targets = full_dataset.data[final_chosen_list], list(np.array(full_dataset.targets)[final_chosen_list])
        

        # partial_dataset = random_split(full_dataset, [5000, len(full_dataset)-5000])[0]
        self.steps_per_epoch = len(full_dataset) // batch_size
        print("steps_per_epoch: ", self.steps_per_epoch)
        self.data_loader = DataLoader(full_dataset, batch_size = batch_size, shuffle=True)



        # keep a reference to the client
        self.client_rrefs = []
        for client_rref in client_rrefs:
            self.client_rrefs.append(client_rref)

        self.generator = Generator(100)
        self.G_opt = DistributedOptimizer(
           optim.Adam, param_rrefs(self.generator), lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6
        )
        # self.G_opt = torch.optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9))

        self.discriminator = Discriminator()
        self.D_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        # if self.use_cuda:
        #     self.discriminator.to(self.device)
            # self.generator.to(self.device)

        # this discriminator imitates the benign client
        self.discriminator_benign = Discriminator()
        self.D_opt_benign = torch.optim.Adam(self.discriminator_benign.parameters(), lr=1e-4, betas=(0.5, 0.9))   
            
        # register generator for each client.
        for client_rref in self.client_rrefs:
            client_rref.remote().register_G(rpc.RRef(self.generator)) 


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

    def sample_generation(self):
        z = Variable(torch.randn(self.batch_size, *self.latent_shape))
        if self.use_cuda:
            z = z.cuda()
        return self.generator(z)


    def sample_total_generation(self):
        z = Variable(torch.randn(10000, *self.latent_shape))
        # if self.use_cuda:
        #     z = z.cuda()
        return self.generator(z)

    def sample_latent(self):
        z = Variable(torch.randn(self.batch_size, *self.latent_shape))
        # if self.use_cuda:
        #     z = z.cuda()
        return z   

    def sample_testing_latent(self):
        z = Variable(torch.randn(1000, *self.latent_shape))
        # if self.use_cuda:
        #     z = z.cuda()
        return self.generator(z)

    def save_gif(input_path, output_path):
        allFrames = []
        for dir in glob.glob(f"{input_path}cifar10-epoch*"):
            print(dir)
            frames = [Image.open(image) for image in glob.glob(f"{dir}/*.jpg")]
            allFrames.extend(frames)
        frame_one = allFrames[0]
        frame_one.save(output_path, format="GIF", append_images=frames,
                save_all=True, duration=200, loop=0)

    def gradient_penalty(self, data, generated_data, gamma=10):
        batch_size = data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1)
        epsilon = epsilon.expand_as(data)


        # if self.use_cuda:
        #     epsilon = epsilon.cuda()

        interpolation = epsilon * data.data + (1 - epsilon) * generated_data.data
        interpolation = Variable(interpolation, requires_grad=True)

        # if self.use_cuda:
        #     interpolation = interpolation.cuda()
        # print("shape of data", data.shape, generated_data.shape, interpolation.shape)
        interpolation_logits = self.discriminator_benign(interpolation)
        grad_outputs = torch.ones(interpolation_logits.size())

        # if self.use_cuda:
        #     grad_outputs = grad_outputs.cuda()

        gradients = autograd.grad(outputs=interpolation_logits,
                                  inputs=interpolation,
                                  grad_outputs=grad_outputs,
                                  create_graph=True,
                                  retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        # here add an epsilon to avoid sqrt error for number close to 0
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12) 
        return gamma * ((gradients_norm - 1) ** 2).mean()

    def train_local_D(self):

        iterloader = iter(self.data_loader)
        try:
            data, _ = next(iterloader)
        except StopIteration:
            iterloader = iter(self.data_loader)
            data, _ = next(iterloader)
            
        generated_data = self.generator(self.sample_latent())
        # generated_data = generated_data.to(device=self.device)
        # data = data.to(self.device)

        grad_penalty = self.gradient_penalty(data, generated_data)
        d_loss = self.discriminator_benign(generated_data).mean() - self.discriminator_benign(data).mean() + grad_penalty
        self.D_opt_benign.zero_grad()
        d_loss.backward()
        self.D_opt_benign.step() 


    def train_D(self):
        
        # iterloader = iter(self.data_loader)
        # try:
        #     data = next(iterloader)
        # except StopIteration:
        #     iterloader = iter(self.data_loader)
        #     data = next(iterloader)

        for i, (data, _) in enumerate(self.data_loader):
            
            generated_data = self.generator(self.sample_latent())
            generated_data = generated_data.to(self.device)
            data = data.to(self.device)

            grad_penalty = self.gradient_penalty(data, generated_data)
            d_loss = self.discriminator(generated_data).mean() - self.discriminator(data).mean() + grad_penalty
            self.D_opt.zero_grad()
            d_loss.backward()
            self.D_opt.step()

        return d_loss.item(), grad_penalty.item()

    def fit(self):

        """
        E: the interval epochs to swap models
        """

        steps_per_epoch_list = []
        for client_rref in self.client_rrefs:
            steps_per_epoch_list.append(client_rref.remote().get_steps_number().to_here())
        self.steps_per_epoch = np.max(steps_per_epoch_list)
        


        # testing_discriminator_token = False
        for j in range(self.epochs):
            print("epoch: ", j)
            testing_discriminator_token = False
            for i in range(self.steps_per_epoch):
                if i % 20 == 0:
                    print("on step: ", i)
                loss_fut_list = []
                for client_rref in self.client_rrefs:
                    # print("D step update")
                    loss_fut = client_rref.rpc_async().train_D()
                    loss_fut_list.append(loss_fut)

                self.train_local_D()

                for id_ in range(len(self.client_rrefs)):
                    loss_d, pen = loss_fut_list[id_].wait()
                    # print("LOSS D: ", loss_d, pen)

                # if i % self.n_critic == 0 and testing_discriminator_token == False:
                if i % self.n_critic == 0:
                    if j % 10 == 0 and j > 0 and testing_discriminator_token == False:
                    # if j % 1 == 0 :
                        print("ignore clients: ", self.ignore_clients)
                        prediction_result_list = [None] * int(len(self.client_rrefs)+2) # +2 for two detector
                        include_client_index_list = list(range(len(self.client_rrefs)))
                        for value in self.ignore_clients:
                            include_client_index_list.remove(value)
                        print("include client index list: ", include_client_index_list)

                        fixed_latent = self.sample_testing_latent()
                        with dist.autograd.context() as G_context:
                            # self.G_opt.zero_grad()
                            loss_g_list = []
                            loss_accumulated = None
                            for idx, client_rref in enumerate(self.client_rrefs):
                                # if client in ignore_clients list, skip the training of G by these clients
                                if idx not in self.ignore_clients:
                                    print("used client: ", idx)
                                    loss_g_list.append(client_rref.rpc_async().loss_G_data(fixed_latent))

                            loss_accumulated, output_value = loss_g_list[0].wait()
                            # output the first included client id and its output value
                            print("prediction_result_list: ", include_client_index_list[0], output_value[0:10])
                            prediction_result_list[include_client_index_list[0]] = output_value
                            for n in range(1, len(loss_g_list)):
                                loss_accumulated_current, output_value_current = loss_g_list[n].wait()
                                loss_accumulated += loss_accumulated_current
                                print("prediction_result_list: ", include_client_index_list[n], output_value_current[0:10])
                                prediction_result_list[include_client_index_list[n]] = output_value_current
                            
                            dist.autograd.backward(G_context, [loss_accumulated])
                            self.G_opt.step(G_context)  

                        # print("print(prediction_result_list) : ", prediction_result_list)
                        if len(self.ignore_clients) != 0:
                            loss_g_list = []
                            for idx_ in self.ignore_clients:
                                print("get output from ignore clients")
                                loss_g_list.append(self.client_rrefs[idx_].rpc_async().loss_G_data(fixed_latent))
                            loss_accumulated_list = []
                            for idx_, loss_g in enumerate(loss_g_list):
                                _, output_value = loss_g.wait()
                                prediction_result_list[self.ignore_clients[idx_]] = output_value
                                print("prediction_result_list: ", self.ignore_clients[idx_], output_value[0:10])
                        # get reference output
                        prediction_result_list[len(self.client_rrefs)] = self.discriminator_benign(fixed_latent).detach().numpy()
                        prediction_result_list[len(self.client_rrefs)+1] = self.discriminator(fixed_latent).detach().numpy()
                        
                        print("prediction_result_list detector benign: ", len(self.client_rrefs), prediction_result_list[len(self.client_rrefs)][0:10])
                        print("prediction_result_list detector fr: ", len(self.client_rrefs)+1, prediction_result_list[len(self.client_rrefs)+1][0:10])

                        # print("print(prediction_result_list) : ", prediction_result_list)
                        # until now, prediction_result_list should contains N list.
                    

                        # self.ignore_clients = []
                        # self.result_matrix = distance_matrix(prediction_result_list, prediction_result_list)
                        # print("result_matrix: ", self.result_matrix)
                        # self.distance_matrix_records.append(self.result_matrix)
                        # result_vector = np.sum(self.result_matrix, axis = 1)
                        # print("result_vector: ", result_vector)
                        # self.distance_matrix_records_sum.append(list(result_vector))

                        # kmeans = KMeans(n_clusters=2, random_state=0).fit(prediction_result_list)
                        # cluster_map = pd.DataFrame()
                        # # cluster_map['data'] = result_vector
                        # cluster_map['cluster'] = kmeans.labels_
                        # cluster_zero = cluster_map[cluster_map.cluster == 0]
                        # cluster_one = cluster_map[cluster_map.cluster == 1]
                        # k_biggest_indices = []
                        # local_discriminator_index = int(len(self.client_rrefs))


                        # if local_discriminator_index in cluster_zero.index:
                        #     k_biggest_indices = list(cluster_zero.index)
                        # else:
                        #     k_biggest_indices = list(cluster_one.index)
                        # k_biggest_indices.remove(local_discriminator_index)
                        # print("kmeans labels and chosen ignore clients: ", kmeans.labels_, k_biggest_indices)
                        # self.ignore_clients_record.append(k_biggest_indices)
                        # self.ignore_clients = k_biggest_indices


                        self.ignore_clients = []
                        self.result_matrix = distance_matrix(prediction_result_list, prediction_result_list)
                        print("result_matrix: ", self.result_matrix)
                        self.distance_matrix_records.append(self.result_matrix)
                        result_vector = np.sum(self.result_matrix, axis = 1)
                        print("result_vector: ", result_vector)
                        self.distance_matrix_records_sum.append(list(result_vector))

                        kmeans = KMeans(n_clusters=2, random_state=0).fit(prediction_result_list)
                        cluster_map = pd.DataFrame()
                        # cluster_map['data'] = result_vector
                        cluster_map['cluster'] = kmeans.labels_
                        cluster_zero = cluster_map[cluster_map.cluster == 0]
                        cluster_one = cluster_map[cluster_map.cluster == 1]
                        k_biggest_indices = []

                        local_benign_discriminator_index = int(len(self.client_rrefs))
                        local_fr_discriminator_index = int(len(self.client_rrefs)+1)

                        if (local_fr_discriminator_index in cluster_zero.index and cluster_zero.shape[0] == 1) or (local_fr_discriminator_index in cluster_one.index and cluster_one.shape[0] == 1):
                            # in that case, all clients are benign.
                            print("no ignore clients, all clients are benign")
                            self.ignore_clients_record.append([])
                            self.ignore_clients = []

                        # in following case, the clients close to benign detector will be judged as benign.
                        if local_benign_discriminator_index in cluster_zero.index:
                            k_biggest_indices = list(cluster_one.index)
                        else:
                            k_biggest_indices = list(cluster_zero.index)
                        if local_fr_discriminator_index in k_biggest_indices:
                            k_biggest_indices.remove(local_fr_discriminator_index)
                        print("kmeans labels and chosen ignore clients: ", kmeans.labels_, k_biggest_indices)
                        self.ignore_clients_record.append(k_biggest_indices)
                        self.ignore_clients = []
                        if len(k_biggest_indices) != len(self.client_rrefs):                           
                            self.ignore_clients = k_biggest_indices




                        testing_discriminator_token = True
                        # fixed_latent = self.sample_testing_latent()
                        # with dist.autograd.context() as G_context:
                        #     # self.G_opt.zero_grad()
                        #     loss_g_list = []
                        #     loss_accumulated = None
                        #     for idx, client_rref in enumerate(self.client_rrefs):
                        #         # if client in ignore_clients list, skip the training of G by these clients
                        #         if idx not in self.ignore_clients:
                        #             print("used client: ", idx)
                        #             loss_g_list.append(client_rref.rpc_async().loss_G_data(fixed_latent))

                        #     loss_accumulated, output_value = loss_g_list[0].wait()
                        #     print("prediction_result_list: ", include_client_index_list[0], output_value[0:10])
                        #     prediction_result_list[include_client_index_list[0]] = output_value
                        #     for n in range(1, len(loss_g_list)):
                        #         loss_accumulated_current, output_value_current = loss_g_list[n].wait()
                        #         loss_accumulated += loss_accumulated_current
                        #         print("prediction_result_list: ", include_client_index_list[n], output_value_current[0:10])
                        #         prediction_result_list[include_client_index_list[n]] = output_value_current
                            
                        #     dist.autograd.backward(G_context, [loss_accumulated])
                        #     self.G_opt.step(G_context)  

                        # # print("print(prediction_result_list) : ", prediction_result_list)
                        # if len(self.ignore_clients) != 0:
                        #     loss_g_list = []
                        #     for idx_ in self.ignore_clients:
                        #         print("get output from ignore clients")
                        #         loss_g_list.append(self.client_rrefs[idx_].rpc_async().loss_G_data(fixed_latent))
                        #     loss_accumulated_list = []
                        #     for idx_, loss_g in enumerate(loss_g_list):
                        #         _, output_value = loss_g.wait()
                        #         prediction_result_list[self.ignore_clients[idx_]] = output_value
                        # # print("print(prediction_result_list) : ", prediction_result_list)
                        # # until now, prediction_result_list should contains N list.
                    

                        # self.ignore_clients = []
                        # self.result_matrix = distance_matrix(prediction_result_list, prediction_result_list)
                        # print("result_matrix: ", self.result_matrix)
                        # self.distance_matrix_records.append(self.result_matrix)
                        # result_vector = np.sum(self.result_matrix, axis = 1)
                        # print("result_vector: ", result_vector)
                        # self.distance_matrix_records_sum.append(list(result_vector))

                        # kmeans = KMeans(n_clusters=2, random_state=0).fit(result_vector.reshape(-1,1))
                        # cluster_map = pd.DataFrame()
                        # cluster_map['data'] = result_vector
                        # cluster_map['cluster'] = kmeans.labels_
                        # cluster_zero = cluster_map[cluster_map.cluster == 0]
                        # cluster_one = cluster_map[cluster_map.cluster == 1]
                        # k_biggest_indices = []
                        # if cluster_one['data'].mean() >= cluster_zero['data'].mean():
                        #     k_biggest_indices = cluster_one.index
                        # else:
                        #     k_biggest_indices = cluster_zero.index

                        # print("kmeans labels and chosen ignore clients: ", kmeans.labels_, k_biggest_indices)
                        # self.ignore_clients_record.append(k_biggest_indices)
                        # self.ignore_clients = k_biggest_indices
                        # testing_discriminator_token = True
                    else: 
                        # if j > 10:    # warm up
                        with dist.autograd.context() as G_context:
                            # self.G_opt.zero_grad()
                            loss_g_list = []
                            loss_accumulated = None
                            for idx, client_rref in enumerate(self.client_rrefs):
                                # if client in ignore_clients list, skip the training of G by these clients
                                if idx not in self.ignore_clients:
                                    print("used client: ", idx)
                                    loss_g_list.append(client_rref.rpc_async().loss_G())

                            loss_accumulated = loss_g_list[0].wait()
                            for n in range(1, len(loss_g_list)):
                                loss_accumulated += loss_g_list[n].wait()
                            
                            dist.autograd.backward(G_context, [loss_accumulated])
                            self.G_opt.step(G_context)
            
            # for record the progress of the image generation
            if (j+1) % 5 == 0: # each 5 epochs, we generate a batch of image to detect the quality
                generation = self.sample_total_generation()
                print("generation: ", generation.shape)
                print("generations shape after transpose: ", generation.shape)
                path = 'data/cifar10-epoch{}'.format(j)

                # Check whether the specified path exists or not
                isExist = os.path.exists(path)
                if not isExist:     
                    # Create a new directory because it does not exist 
                    os.makedirs(path)
                    print("The new directory is created!")

                for idx, img in enumerate(generation):
                    save_image(img, 'data/cifar10-epoch{}/{:05d}.jpg'.format(j,idx))


            # for swapping models
            if  len(self.client_rrefs) > 1 and j > 0 and j % 10 == 0: # j%10 means swapping models every 10 epochs
                print("in swap")
                list_to_choose = list(np.arange(len(self.client_rrefs)))
                for index in range(len(self.client_rrefs)):
                    print("index: ", index)
                    if index in list_to_choose:
                        if len(list_to_choose) == 1:
                            list_to_choose.remove(index)
                            list_chosen = list(np.arange(len(self.client_rrefs)))
                            list_chosen.remove(index)
                            random_index = np.random.choice(list_chosen, 1, replace=False)[0]
                            self.attempted_switch.append([j, index, random_index])
                            # print("indices for switching: ", index, random_index)
                            if len(self.result_matrix) != 0 and swap_decision(index, random_index, self.result_matrix):
                                # print("chosen random index for last swapper: ", index, random_index)
                                self.success_switch.append([j, index, random_index])
                                state_dic_temp = self.client_rrefs[random_index].rpc_sync().get_discriminator_weights()
                                self.client_rrefs[random_index].rpc_sync().set_discriminator_weights(self.client_rrefs[index].rpc_sync().get_discriminator_weights())
                                self.client_rrefs[index].rpc_sync().set_discriminator_weights(state_dic_temp)
                        else:
                            list_to_choose.remove(index)
                            random_index = np.random.choice(list_to_choose, 1, replace=False)[0]
                            list_to_choose.remove(random_index)
                            self.attempted_switch.append([j, index, random_index])
                            print("indices for switching: ", index, random_index)
                            print("self.result matrix: ", self.result_matrix)
                            if len(self.result_matrix) != 0 and swap_decision(index, random_index, self.result_matrix):
                                # print("chosen random index for swapping: ", index, random_index)
                                self.success_switch.append([j, index, random_index])
                                state_dic_temp = self.client_rrefs[random_index].rpc_sync().get_discriminator_weights()
                                self.client_rrefs[random_index].rpc_sync().set_discriminator_weights(self.client_rrefs[index].rpc_sync().get_discriminator_weights())
                                self.client_rrefs[index].rpc_sync().set_discriminator_weights(state_dic_temp)



            # if j % 10 == 0 and j > 0:
            # # if j % 1 == 0:    
            #     self.ignore_clients = []
            #     test_result = []
            #     prediction_result_list = []
            #     testing_latent = self.sample_testing_latent()
            #     for client_rref in self.client_rrefs:
            #         loss_fut = client_rref.rpc_async().evalute_D(testing_latent)
            #         test_result.append(loss_fut)

            #     for id_ in range(len(self.client_rrefs)):
            #         return_result = test_result[id_].wait()
            #         # print(id_, return_result[0:10], len(return_result))
            #         prediction_result_list.append(return_result)

            #     result_matrix = distance_matrix(prediction_result_list, prediction_result_list)
            #     print("result_matrix: ", result_matrix)
            #     self.distance_matrix_records.append(result_matrix)
            #     result_vector = np.sum(result_matrix, axis = 1)
            #     print("result_vector: ", result_vector)
            #     self.distance_matrix_records_sum.append(list(result_vector))

            #     kmeans = KMeans(n_clusters=2, random_state=0).fit(result_vector.reshape(-1,1))
            #     cluster_map = pd.DataFrame()
            #     cluster_map['data'] = result_vector
            #     cluster_map['cluster'] = kmeans.labels_
            #     cluster_zero = cluster_map[cluster_map.cluster == 0]
            #     cluster_one = cluster_map[cluster_map.cluster == 1]
            #     k_biggest_indices = []
            #     if cluster_one['data'].mean() >= cluster_zero['data'].mean():
            #         k_biggest_indices = cluster_one.index
            #     else:
            #         k_biggest_indices = cluster_zero.index
            #     # k_biggest_indices = sorted(range(len(result_vector)), key = lambda sub: result_vector[sub], reverse=True)[:3]

            #     print("kmeans labels and chosen ignore clients: ", kmeans.labels_, k_biggest_indices)
            #     self.ignore_clients_record.append(k_biggest_indices)
            #     self.ignore_clients = k_biggest_indices

        # save Discriminator models
        Path("savedModels").mkdir(parents=False, exist_ok=True)
        for id, client in enumerate(self.client_rrefs):
            PATH = 'savedModels/Discriminator{}_state_dict_model.pt'.format(id)
            torch.save(client.rpc_sync().get_discriminator_weights(), PATH)

        with open("DISTANCE_MATRIX_5000ROWS_5CLIENT_0attacker_LIST_ROUND2.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.distance_matrix_records) 

        with open("DISTANCE_MATRIX_5000ROWS_5CLIENT_0attacker_SUM_ROUND2.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.distance_matrix_records_sum) 

        with open("IGNORE_CLIENTS_5000ROWS_5CLIENT_0attacker_ROUND2.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.ignore_clients_record)        

        with open("ATTEMPTED_SWITCH_5000ROWS_5CLIENT_0attacker_ROUND2.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.attempted_switch)                    

        with open("SUCCESS_SWITCH_5000ROWS_5CLIENT_0attacker_ROUND2.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.success_switch)                    

            



class MDGANClient():
    """
    This is the class that encapsulates the functions that need to be run on the client side
    Despite the name, this source code only needs to reside on the server and will be executed via RPC.
    """

    def __init__(self, dataset, epochs, use_cuda, batch_size, **kwargs):
        self.epochs = epochs
        self.latent_shape = [100, 1, 1]
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        if dataset == 'cifar10':
            train_dataset = CIFAR10(root='data/cifar10', train=True, download=False, transform=transform)
            test_dataset = CIFAR10(root='data/cifar10', train=False, download=False, transform=transform)


        if dataset == 'mnist':
            train_dataset = MNIST(root='data/mnist', train=True, download=False, transform=transform)
            test_dataset = MNIST(root='data/mnist', train=False, download=False, transform=transform)

        if dataset == 'fashion':
            train_dataset = FashionMNIST(root='data/fashion', train=True, download=False, transform=transform)
            test_dataset = FashionMNIST(root='data/fashion', train=False, download=False, transform=transform)

        full_dataset = ConcatDataset([train_dataset, test_dataset])
        self.steps_per_epoch = full_dataset.shape[0] // batch_size
        self.data_loader = DataLoader(full_dataset, batch_size = batch_size, shuffle=True)
        self.discriminator = Discriminator()
        self.D_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        if self.use_cuda:
            self.discriminator.cuda()


    def send_client_refs(self):
        """Send a reference to the discriminator (for future RPC calls) and the conditioner, transformer and steps/epoch"""
        return rpc.RRef(self.discriminator)


    def register_G(self, G_rref):
        """Receive a reference to the generator (for future RPC calls)"""
        self.G_rref = G_rref
        
    def sample_latent(self):
        z = Variable(torch.randn(self.batch_size, *self.latent_shape))
        # if self.use_cuda:
        #     z = z.cuda()
        return z

    def get_steps_number(self):
        return self.steps_per_epoch

    def get_discriminator_weights(self):
        return self.discriminator.state_dict()

    def reset_on_cuda(self):
        if self.device.type != 'cpu':
            self.discriminator.to(self.device)

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

    def train_D(self):
        
        # iterloader = iter(self.data_loader)
        # try:
        #     data = next(iterloader)
        # except StopIteration:
        #     iterloader = iter(self.data_loader)
        #     data = next(iterloader)

        for i, (data, _) in enumerate(self.data_loader):
            
            generated_data = self.G_rref.remote().forward(self.sample_latent().cpu()).to_here()
            generated_data = generated_data.to(device=self.device)
            data = data.to(self.device)

            grad_penalty = self.gradient_penalty(data, generated_data)
            d_loss = self.discriminator(generated_data).mean() - self.discriminator(data).mean() + grad_penalty
            self.D_opt.zero_grad()
            d_loss.backward()
            self.D_opt.step()

        return d_loss.item(), grad_penalty.item()

    

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
        loss_g_list = []
        #for i, (_, _) in enumerate(self.data_loader): # ? take the length of data
        for _ in range(self.steps_per_epoch):
            generated_data = self.G_rref.remote().forward(self.sample_latent().cpu()).to_here()
            loss_g = -self.discriminator(generated_data).mean()
            if self.device.type != 'cpu':
                loss_g_list.append(loss_g.cpu())
            else:
                loss_g_list.append(loss_g)
        return sum(loss_g_list)

    def evalute_D(self, testing_latent):
        if self.device.type != 'cpu':
            return torch.max(self.discriminator(testing_latent), 1)[1].cpu()
        else:
            return torch.max(self.discriminator(testing_latent), 1)[1]




def run(rank, world_size, ip, port, dataset, epochs, use_cuda, batch_size, n_critic):
    # set environment information
    os.environ["MASTER_ADDR"] = ip
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    # print("number of epochs before initialization: ", epochs)
    # print("world size: ", world_size, f"tcp://{ip}:{port}")
    if rank == 0:  # this is run only on the server side
        rpc.init_rpc(
            "server",
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                num_worker_threads=8, rpc_timeout=120, init_method=f"tcp://{ip}:{port}", _transports=["uv"]
            ),
        )
        print("Server joined")
        clients = []
        for worker in range(world_size-1):
            clients.append(rpc.remote("client"+str(worker+1), MDGANClient, kwargs=dict(dataset=dataset, epochs = epochs, use_cuda = use_cuda, batch_size=batch_size)))
            print("register remote client"+str(worker+1), clients[0])

        synthesizer = MDGANServer(clients, epochs, use_cuda, batch_size, n_critic)
        synthesizer.fit()

    elif rank != 0:
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
    parser.add_argument("-rank", type=int, default=0)
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
