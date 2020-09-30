import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


class PsiCompressor(nn.Module):
    def __init__(self, input_param, out_dim):
        super().__init__()
        self.fc = nn.Linear(input_param, out_dim)
        self.out_dim = out_dim
        
    def forward(self, psi):
        return torch.tanh(self.fc(psi))


class Net(nn.Module):
    def __init__(self, out_dim, psi_dim, hidden_dim=100, X_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(X_dim + psi_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)    
        
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0.0)
        
    def forward(self, params):
        """
            Generator takes a vector of noise and produces sample
        """
        # psi_embedding = self.pc(params[:, :self.psi_dim])
        # z = torch.cat([z, psi_embedding, params[:, self.psi_dim:]], dim=1)
        h1 = torch.tanh(self.fc1(params))
        h2 = torch.tanh(self.fc2(h1))
        h3 = F.leaky_relu(self.fc4(h2))
        y_gen = self.fc3(h3)
        return y_gen        


# Idea is kind of taken from https://arxiv.org/pdf/1805.08318.pdf and applied to conditions
class Attention(nn.Module):
    def __init__(self, psi_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self._psi_dim = psi_dim

        self.f_net = nn.Conv1d(1, hidden_dim, 1, stride=1, padding=0, bias=False)
        self.g_net = nn.Conv1d(1, hidden_dim, 1, stride=1, padding=0, bias=False)
        self.h_net = nn.Conv1d(1, hidden_dim, 1, stride=1, padding=0, bias=False)
        self.v_net = nn.Conv1d(hidden_dim, 1, 1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.Tensor([0.]), requires_grad=True)

    def forward(self, params):
        params_psi = params[:, :self._psi_dim].unsqueeze(1)
        attention_map = self.softmax(torch.bmm(self.f_net(params_psi).transpose(1, 2), self.g_net(params_psi)))
        self_attention = self.v_net(torch.bmm(attention_map, self.h_net(params_psi).transpose(1, 2)).transpose(1, 2))
        psi_with_attention = (self.gamma * self_attention + params_psi)[:, 0, :]
        return torch.cat([psi_with_attention, params[:, self._psi_dim:]], dim=1)


class SimpleAttention(nn.Module):
    def __init__(self, psi_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self._psi_dim = psi_dim

        self._attention_net = nn.Sequential(
            nn.Linear(psi_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, psi_dim),
            nn.Softmax(dim=1)
        )
        self.gamma = nn.Parameter(torch.Tensor([0.]), requires_grad=True)

    def forward(self, params):
        psi_with_attention = params[:, :self._psi_dim] * self._attention_net(params[:, :self._psi_dim])
        return torch.cat([psi_with_attention, params[:, self._psi_dim:]], dim=1)


class Generator(nn.Module):
    def __init__(self, noise_dim, out_dim, psi_dim, hidden_dim=100, x_dim=1, attention_net=None):
        super(Generator, self).__init__()
        
        self.fc1 = nn.Linear(noise_dim + x_dim + psi_dim, hidden_dim)
        # nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.constant_(self.fc1.bias, 0.0)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.constant_(self.fc2.bias, 0.0)
        
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        # nn.init.xavier_normal_(self.fc3.weight)
        # nn.init.constant_(self.fc3.bias, 0.0)
        
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        # nn.init.xavier_normal_(self.fc4.weight)
        # nn.init.constant_(self.fc4.bias, 0.0)
        
        # self.pc = psi_compressor
        self.psi_dim = psi_dim
        self.attention_net = attention_net

    def forward(self, z, params):
        """
            Generator takes a vector of noise and produces sample
        """
        # psi_embedding = self.pc(params[:, :self.psi_dim])
        # z = torch.cat([z, psi_embedding, params[:, self.psi_dim:]], dim=1)
        if self.attention_net:
            params = self.attention_net(params)
        z = torch.cat([z, params], dim=1)
        h1 = torch.tanh(self.fc1(z))
        h4 = torch.tanh(self.fc4(h1))
        h2 = F.leaky_relu(self.fc2(h4))
        y_gen = self.fc3(h2)
        return y_gen


class Discriminator(nn.Module):
    def __init__(self,
                 in_dim,
                 psi_dim,
                 hidden_dim=100,
                 output_logits=False,
                 x_dim=1,
                 output_dim=1,
                 attention_net=None):
        super(Discriminator, self).__init__()
        self.output_logits = output_logits

        self.fc1 = nn.Linear(in_dim + x_dim + psi_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)
        
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0.0)
        
        # self.pc = psi_compressor
        self.psi_dim = psi_dim
        self.attention_net = attention_net

    def forward(self, x, params):
        if self.attention_net:
            params = self.attention_net(params)
        x = torch.cat([x, params], dim=1)
        # psi_embedding = self.pc(params[:, :self.psi_dim])
        # x = torch.cat([x, psi_embedding, params[:, self.psi_dim:]], dim=1)
        h1 = torch.tanh(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        # h3 = F.leaky_relu(self.fc3(h2))
        if self.output_logits:
            return self.fc4(h2)
        else:
            return torch.sigmoid(self.fc4(h2))


class GANLosses(object):
    def __init__(self, task):
        self.TASK = task

    def g_loss(self, discrim_output, discrim_output_gen_prime=None, discrim_output_real=None):
        eps = 1e-10
        if self.TASK == 'KL':
            loss = torch.log(1 - discrim_output + eps).mean()
        elif self.TASK == 'REVERSED_KL':
            loss = - torch.log(discrim_output + eps).mean()
        elif self.TASK == 'WASSERSTEIN':
            loss = - discrim_output.mean()
        elif self.TASK == "CRAMER":
            # loss = (torch.norm(discrim_output_real - discrim_output, dim=1) + \
            #        torch.norm(discrim_output_real - discrim_output_gen_prime, dim=1) - \
            #        torch.norm(discrim_output - discrim_output_gen_prime, dim=1)).mean()
            loss =   (torch.norm(discrim_output_real - discrim_output_gen_prime, dim=1) -
                      torch.norm(discrim_output_real, dim=1) -
                      torch.norm(discrim_output - discrim_output_gen_prime, dim=1) +
                      torch.norm(discrim_output, dim=1)).mean()
        return loss

    def d_loss(self, discrim_output_gen, discrim_output_real, discrim_output_gen_prime=None):
        eps = 1e-10
        if self.TASK in ['KL', 'REVERSED_KL']:
            loss = - torch.log(discrim_output_real + eps).mean() - torch.log(1 - discrim_output_gen + eps).mean()
        elif self.TASK == 'WASSERSTEIN':
            loss = - (discrim_output_real.mean() - discrim_output_gen.mean())
        elif self.TASK == "CRAMER":
            loss = - (torch.norm(discrim_output_real - discrim_output_gen_prime, dim=1) -
                      torch.norm(discrim_output_real, dim=1) -
                      torch.norm(discrim_output_gen - discrim_output_gen_prime, dim=1) +
                      torch.norm(discrim_output_gen, dim=1)).mean()
        return loss

    def calc_gradient_penalty(self, discriminator, data_gen, inputs_batch, inp_data, lambda_reg=.1, data_gen_prime=None):
        device = data_gen.device
        alpha = torch.rand(inp_data.shape[0], 1).to(device)
        dims_to_add = len(inp_data.size()) - 2
        for i in range(dims_to_add):
            alpha = alpha.unsqueeze(-1)

        interpolates = (alpha * inputs_batch + ((1 - alpha) * data_gen)).to(device)
        interpolates.requires_grad_(True)
        disc_interpolates = discriminator(interpolates, inp_data)
        if self.TASK == "WASSERSTEIN":
            gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        elif self.TASK == "CRAMER":
            f_val = torch.norm(discriminator(data_gen_prime, inp_data) - disc_interpolates, dim=1) - \
                    torch.norm(disc_interpolates, dim=1)
            gradients = torch.autograd.grad(outputs=f_val, inputs=interpolates,
                                            grad_outputs=torch.ones(f_val.size()).to(device),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_norm_diff = gradients.norm(2, dim=1) - 1
        grad_norm_diff[grad_norm_diff < 0] = 0
        gradient_penalty = (grad_norm_diff ** 2).mean() * lambda_reg
        return gradient_penalty

    def calc_zero_centered_GP(self, discriminator, data_gen, inputs_batch, inp_data, gamma_reg=.1):
        # TODO: data_gen is not used!
        device = inputs_batch.device
        local_input = inp_data.clone().detach().requires_grad_(True)
        disc_interpolates = discriminator(local_input, inputs_batch)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=local_input,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        return gamma_reg / 2 * (gradients.norm(2, dim=1) ** 2).mean() 
