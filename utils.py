import numpy as np
from tqdm import trange
import torch
import matplotlib.pyplot as plt
from pyro import distributions as dist
from model import OptLoss
from copy import deepcopy
import lhsmdu
import matplotlib.patches as patches

my_cmap = plt.cm.jet
my_cmap.set_under('white')


def sample_noise(N, NOISE_DIM):
    return np.random.normal(size=(N, NOISE_DIM)).astype(np.float32)


def iterate_minibatches(X, batchsize, y=None):
    perm = np.random.permutation(X.shape[0])
    
    for start in range(0, X.shape[0], batchsize):
        end = min(start + batchsize, X.shape[0])
        if y is None:
            yield X[perm[start:end]]
        else:
            yield X[perm[start:end]], y[perm[start:end]]


def generate_data(y_sampler, n_samples, mu_range=(-5, 5), mu_dim=1, x_dim=1):
    # mus = torch.empty([n_samples, mu_dim]).uniform_(*mu_range).to(device)
    mus = torch.randint(*mu_range, [n_samples, mu_dim], dtype=torch.float32) # .to(device)
    xs = y_sampler.x_dist.sample(torch.Size([n_samples, x_dim])) # .to(device)

    y_sampler.make_condition_sample({'mu': mus, 'X':xs})
    
    data = y_sampler.condition_sample().detach() # .to(device)
    return data.reshape(-1, 1), torch.cat([mus, xs], dim=1)


def generate_local_data(y_sampler, n_samples_per_dim, step, current_psi, x_dim=1, std=0.1):
    xs = y_sampler.x_dist.sample(torch.Size([n_samples_per_dim * 2 * current_psi.shape[1] + n_samples_per_dim, x_dim])) # .to(device)

    mus = torch.empty((xs.shape[0], current_psi.shape[1])) # .to(device)

    iterator = 0
    for dim in range(current_psi.shape[1]):
        for dir_step in [-step, step]:
            random_mask = torch.torch.randn_like(current_psi)
            random_mask[0, dim] = 0
            new_psi = current_psi + random_mask * std
            new_psi[0, dim] += dir_step

            mus[iterator:
                iterator + n_samples_per_dim, :] = new_psi.repeat(n_samples_per_dim, 1)
            iterator += n_samples_per_dim

    mus[iterator: iterator + n_samples_per_dim, :] = current_psi.repeat(n_samples_per_dim, 1).clone().detach()
            
    y_sampler.make_condition_sample({'mu': mus, 'X': xs})
    data = y_sampler.condition_sample().detach().to(device)
    return data.reshape(-1, 1), torch.cat([mus, xs], dim=1)


def generate_local_data_lhs(y_sampler, n_samples_per_dim, step, current_psi, x_dim=1, n_samples=2):
    xs = y_sampler.x_dist.sample(torch.Size([n_samples_per_dim * n_samples, x_dim]))  # .to(device)

    mus = torch.empty((len(xs), len(current_psi)))  # .to(device)
    mus = torch.tensor(lhsmdu.sample(len(current_psi),
                                     n_samples, 
                                     randomSeed=np.random.randint(1e5)).T).float()  # .to(device)

    mus = step * (mus * 2 - 1) + current_psi  # .to(device)
    mus = mus.repeat(1, n_samples_per_dim).reshape(-1, len(current_psi))
    y_sampler.make_condition_sample({'mu': mus, 'X': xs})
    data = y_sampler.condition_sample().detach()  # .to(device)
    return data.reshape(-1, 1), torch.cat([mus, xs], dim=1)


class DistPlotter(object):
    def __init__(self, y_sampler, generator, noise, device, mu_dim=1, x_dim=1):
        self.y_sampler = y_sampler
        self.generator = generator
        self.fixed_noise = noise
        self.device = device
        self.mu_dim = mu_dim
        self.x_dim = x_dim

    def draw_conditional_samples(self, mu_range):
        f = plt.figure(figsize=(21, 16))
        
        mu = dist.Uniform(*mu_range).sample([16, self.mu_dim])
        x = self.y_sampler.x_dist.sample([16, self.x_dim])
        
        for index in range(16):
            plt.subplot(4, 4, index + 1)
            mu_s = mu[index, :].repeat(len(self.fixed_noise), 1).to(self.device)
            x_s = x[index, :].repeat(len(self.fixed_noise), 1).to(self.device)
            self.y_sampler.make_condition_sample({'mu': mu_s, 'X':x_s})
            data = self.y_sampler.condition_sample().detach().cpu().numpy()
            
            plt.hist(data, bins=100, density=True, label='true');
            plt.hist(self.generator(self.fixed_noise, torch.cat([mu_s, x_s], dim=1)).detach().cpu().numpy(),
                     bins=100, color='g', density=True, alpha=0.5, label='gan');
            plt.grid()
            plt.legend()
            plt.ylabel("x={}".format(x[index, :].cpu().numpy()), fontsize=15)
            plt.title("mu={}".format(mu[index, :].cpu().numpy()), fontsize=15)            
        return f
     
    def draw_mu_samples(self, mu_range, noise_size=1000, n_samples=1000):
        f = plt.figure(figsize=(21, 16))
        mu = dist.Uniform(*mu_range).sample([16, self.mu_dim])
        for index in range(16):
            plt.subplot(4, 4, index + 1)
            noise = torch.Tensor(sample_noise(self.fixed_noise.shape[0], self.fixed_noise.shape[1])).to(self.device)
            mu_s = mu[index, :].repeat(self.fixed_noise.shape[0], 1).to(self.device)
            x_s = self.y_sampler.x_dist.sample([len(mu_s), self.x_dim]).to(self.device)
            self.y_sampler.make_condition_sample({'mu': mu_s, 'X':x_s})

            plt.hist(self.y_sampler.condition_sample().cpu().numpy(), bins=100, density=True, label='true');
            plt.hist(self.generator(noise, torch.cat([mu_s, x_s], dim=1)).detach().cpu().numpy(),
                     bins=100, color='g', density=True, alpha=0.5, label='gan');    
            plt.grid()
            plt.legend()
            plt.title("mu={}".format(mu[index, :].cpu().numpy()), fontsize=15);
        return f
            
    def draw_X_samples(self, x_range):
        f = plt.figure(figsize=(21,16))
        x = dist.Uniform(*x_range).sample([12, self.x_dim])
        for index in range(12):
            plt.subplot(4,3, index + 1)
            x_s = x[index, :].repeat(len(self.fixed_noise), 1).to(self.device)
            mu_s = self.y_sampler.mu_dist.sample(torch.Size([len(x_s), self.mu_dim])).to(self.device)
            self.y_sampler.make_condition_sample({'mu': mu_s, 'X':x_s})

            plt.hist(self.y_sampler.condition_sample().cpu().numpy(), bins=100, density=True, label='true');
            plt.hist(self.generator(self.fixed_noise, torch.cat([mu_s,x_s],dim=1)).detach().cpu().numpy(),
                     bins=100, color='g', density=True, alpha=0.5, label='gan');
            plt.grid()
            plt.legend()
            plt.title("x={}".format(x[index, :].cpu().numpy()), fontsize=15)
        return f
    
    def draw_mu_2d_samples(self, mu_range, noise_size=1000):
        my_cmap = plt.cm.jet
        my_cmap.set_under('white')
        mu = dist.Uniform(*mu_range).sample([5000, 2]).to(self.device)
        
        y = np.zeros([len(mu), 1])
        
        for i in range(len(mu)):
            noise = torch.Tensor(sample_noise(noise_size, self.fixed_noise.shape[1])).to(self.device)
            mu_r = mu[i, :].reshape(1,-1).repeat(noise_size, 1).to(self.device)
            x_r = self.y_sampler.x_dist.sample(torch.Size([len(mu_r), 1])).to(self.device)
            y[i, 0] = self.generator(noise, torch.cat([mu_r,x_r],dim=1)).mean().item()

        f = plt.figure(figsize=(12,6))
        mu = mu.cpu().numpy()
        plt.scatter(mu[:,0], mu[:, 1], c=y[:,0], cmap=my_cmap)
        plt.colorbar()
        return f
    
    def plot_means_diff(self, mu_range, x_range):
        means_diff = []
        for index, mu in enumerate(torch.arange(*mu_range, 1)):
            t_means = []
            g_means = []
            for x in torch.arange(*x_range, 0.5):
                # plt.subplot(5, 4, index + 1)
                mu_s = mu.float().reshape(-1,1).repeat(self.fixed_noise.shape[0], 1).to(self.device)
                noise = torch.Tensor(sample_noise(self.fixed_noise.shape[0], self.fixed_noise.shape[1])).to(self.device)
                x_s = x.float().reshape(-1,1).repeat(self.fixed_noise.shape[0], 1).to(self.device)
                y_samples = self.generator(noise, torch.cat([mu_s, x_s], dim=1)).cpu().detach().numpy()
                self.y_sampler.make_condition_sample({'mu': mu_s, 'X':x_s})
                t_means.append(np.mean(y_samples))
                g_means.append(self.y_sampler.condition_sample().cpu().numpy().mean())
            if index == 10:
                f = plt.figure(figsize=(12,6))
                plt.scatter(np.arange(*x_range, 0.5), t_means, label='g')
                plt.scatter(np.arange(*x_range, 0.5), g_means, label='t')
                plt.legend()
                plt.grid()
            means_diff.append((np.array(g_means) - np.array(t_means)).mean())
        g = plt.figure(figsize=(12,6))
        plt.scatter(np.arange(*mu_range, 1), means_diff)
        plt.xlabel(f"$\mu$", fontsize=19)
        plt.ylabel("means_diff")
        plt.grid()
        return f, g
    
    def draw_grads_and_losses(self, current_psi, psi_size=2000, average_size=1000, step=1):
        psi_range = (current_psi - 3 * step, current_psi + 3 * step)        
        
        psi_grid = dist.Uniform(*psi_range).sample([psi_size]).to(self.device)
        x = self.y_sampler.x_dist.sample([average_size * psi_size, 1]).to(self.device)

        psi = psi_grid.repeat(1, average_size).view(-1, 2)
        psi.requires_grad = True
        self.y_sampler.make_condition_sample({"mu": psi, "X": x})

        data_gen = self.y_sampler.condition_sample()
        true_loss = OptLoss.SigmoidLoss(data_gen, 5, 10).view(-1, average_size).mean(dim=1)
        true_loss.sum().backward(retain_graph=True)
        true_grads = psi.grad.view(-1, 1).view(psi_size, average_size, 2).mean(dim=1)
        true_grads = true_grads.detach().cpu().numpy()
        psi.grad.zero_()


        data_gen = self.generator(self.fixed_noise, torch.cat([psi, x], dim=1))
        #data_gen = self.generator(torch.cat([psi, x], dim=1))
        gan_loss = OptLoss.SigmoidLoss(data_gen, 5, 10).view(-1, average_size).mean(dim=1)
        gan_loss.sum().backward(retain_graph=False)
        gan_grads = psi.grad.view(-1, 1).view(psi_size, average_size, 2).mean(dim=1)
        gan_grads = gan_grads.detach().cpu().numpy()
        psi.grad.zero_()
        
        f = plt.figure(figsize=(16,8))

        plt.subplot(1,2,1)
        plt.quiver(psi_grid[:, 0].cpu().detach().cpu().numpy(), 
                   psi_grid[:, 1].cpu().detach().cpu().numpy(),
                   -true_grads[:, 0], 
                   -true_grads[:, 1],
                   np.linalg.norm(true_grads,axis=1),
                   cmap=my_cmap)
        plt.colorbar()
        plt.xlabel(f"$\psi_1$", fontsize=19)
        plt.ylabel(f"$\psi_2$", fontsize=19)
        plt.title("True grads", fontsize=15)

        plt.subplot(1,2,2)
        plt.quiver(psi_grid[:, 0].cpu().detach().cpu().numpy(), 
                   psi_grid[:, 1].cpu().detach().cpu().numpy(),
                   -gan_grads[:, 0],
                   -gan_grads[:, 1],
                   np.linalg.norm(gan_grads,axis=1),
                   cmap=my_cmap)
        plt.colorbar()
        plt.xlabel(f"$\psi_1$", fontsize=19)
        plt.ylabel(f"$\psi_2$", fontsize=19)
        plt.title("GAN grads", fontsize=15)
        
        g = plt.figure(figsize=(16, 8))

        ax = plt.subplot(1,2,1)
        plt.scatter(psi_grid[:, 0].cpu().detach().cpu().numpy(), 
                    psi_grid[:,1].cpu().detach().cpu().numpy(),
                    c=true_loss.cpu().detach().numpy(),
                    cmap=my_cmap)
        plt.colorbar()
        plt.xlabel(f"$\psi_1$", fontsize=19)
        plt.ylabel(f"$\psi_2$", fontsize=19)
        plt.title("True loss", fontsize=15)
        rect = patches.Rectangle(current_psi - step, step * 2, step * 2,linewidth=3,edgecolor='black',facecolor='none')
        ax.add_patch(rect)

        ax = plt.subplot(1,2,2)
        plt.scatter(psi_grid[:, 0].cpu().detach().cpu().numpy(), 
                   psi_grid[:,1].cpu().detach().cpu().numpy(), 
                   c=gan_loss.cpu().detach().numpy(),
                   cmap=my_cmap)
        plt.colorbar()
        plt.xlabel(f"$\psi_1$", fontsize=19)
        plt.ylabel(f"$\psi_2$", fontsize=19)
        plt.title("GAN loss", fontsize=15)
        rect = patches.Rectangle(current_psi - step, step * 2, step * 2,linewidth=3,edgecolor='black',facecolor='none')
        ax.add_patch(rect)        
        return f, g