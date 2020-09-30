from base_model import BaseConditionalGenerationOracle
from gan_nets import Generator, Discriminator, Attention, SimpleAttention
from gan_nets import GANLosses
import torch
import torch.utils.data as pytorch_data_utils
from tqdm import trange, tqdm


class GANModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 y_model: BaseConditionalGenerationOracle,
                 noise_dim: int,
                 psi_dim: int,
                 y_dim: int,
                 x_dim: int,
                 batch_size: int,
                 task: str,
                 epochs: int,
                 lr: float,
                 iters_discriminator: int,
                 iters_generator: int,
                 grad_penalty: bool = False,
                 zero_centered_grad_penalty: bool = False,
                 instance_noise_std: float = None,
                 logger=None,
                 burn_in_period=None,
                 averaging_coeff=None,
                 dis_output_dim=1,
                 attention_net_size=None,
                 gp_reg_coeff=None):
        super(GANModel, self).__init__(y_model=y_model, x_dim=x_dim, psi_dim=psi_dim, y_dim=y_dim)
        if task in ['WASSERSTEIN', "CRAMER"]:
            output_logits = True
        else:
            output_logits = False
        self._task = task
        self._noise_dim = noise_dim
        self._psi_dim = psi_dim
        self._y_dim = y_dim
        self._x_dim = x_dim
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._grad_penalty = grad_penalty
        self._zero_centered_grad_penalty = zero_centered_grad_penalty
        self._instance_noise_std = instance_noise_std
        self._iters_discriminator = iters_discriminator
        self._iters_generator = iters_generator
        self._ganloss = GANLosses(task=task)
        self.lambda_reg = gp_reg_coeff

        if attention_net_size:
            self.attention_net = Attention(psi_dim=self._psi_dim, hidden_dim=attention_net_size)
        else:
            self.attention_net = None
        self._generator = Generator(noise_dim=self._noise_dim,
                                    out_dim=self._y_dim,
                                    psi_dim=self._psi_dim,
                                    x_dim=self._x_dim, attention_net=self.attention_net)
        self._discriminator = Discriminator(in_dim=self._y_dim,
                                            output_logits=output_logits,
                                            psi_dim=self._psi_dim,
                                            x_dim=self._x_dim,
                                            output_dim=dis_output_dim, attention_net=None)

        self.logger = logger
        self.burn_in_period = burn_in_period
        self.averaging_coeff = averaging_coeff
        self.gen_average_weights = []
        self._cond_mean = torch.zeros(self._x_dim + self._psi_dim).float().to(y_model._device)
        self._cond_std = torch.ones(self._x_dim + self._psi_dim).float().to(y_model._device)
        self._y_mean = torch.zeros(self._y_dim).float().to(y_model._device)
        self._y_std = torch.ones(self._y_dim).float().to(y_model._device)

    @staticmethod
    def instance_noise(data, std):
        return data + torch.randn_like(data) * data.std(dim=0) * std

    def loss(self, y, condition):
        # y = (y - self._y_mean) / self._y_std
        # condition = (condition - self._cond_mean) / self._cond_std
        return self._discriminator(y, condition)

    def fit(self, y, condition, weights=None):
        self._cond_mean = condition.mean(0).detach().clone()
        self._cond_std = condition.std(0).detach().clone()
        self._y_mean = y.mean(0).detach().clone()
        self._y_std = y.std(0).detach().clone()


        g_optimizer = torch.optim.Adam(self._generator.parameters(), lr=self._lr, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(self._discriminator.parameters(), lr=self._lr, betas=(0.5, 0.999))

        y = (y - self._y_mean) / self._y_std
        condition = (condition - self._cond_mean) / self._cond_std
        condition[torch.isnan(condition)] = 0
        #dataset = torch.utils.data.TensorDataset(y_normed, cond_normed)
        dataset = torch.utils.data.TensorDataset(y, condition)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self._batch_size,
                                                 shuffle=True)

        for epoch in trange(self._epochs):
            dis_epoch_loss = []
            gen_epoch_loss = []
            for y_batch, cond_batch in dataloader:
                # print(y_batch.shape, cond_batch.shape)
                for _ in range(self._iters_discriminator):
                    y_gen = self.generate(condition=cond_batch, normalise=False)
                    if self._instance_noise_std:
                        y_batch = self.instance_noise(y_batch, self._instance_noise_std)
                        y_gen = self.instance_noise(y_gen, self._instance_noise_std)

                    if self._task == "CRAMER":
                        y_gen_prime = self.generate(condition=cond_batch, normalise=False)
                        if self._instance_noise_std:
                            y_gen_prime = self.instance_noise(y_gen_prime, self._instance_noise_std)
                        loss = self._ganloss.d_loss(self.loss(y_gen, cond_batch),
                                                    self.loss(y_batch, cond_batch),
                                                    self.loss(y_gen_prime, cond_batch))
                        if self._grad_penalty:
                            loss += self._ganloss.calc_gradient_penalty(self._discriminator,
                                                                        y_gen.data,
                                                                        y_batch.data,
                                                                        cond_batch.data,
                                                                        data_gen_prime=y_gen_prime.data,
                                                                        lambda_reg=self.lambda_reg)
                    else:
                        loss = self._ganloss.d_loss(self.loss(y_gen, cond_batch),
                                                    self.loss(y_batch, cond_batch))
                        if self._grad_penalty:
                            loss += self._ganloss.calc_gradient_penalty(self._discriminator,
                                                                        y_gen.data,
                                                                        y_batch.data,
                                                                        cond_batch.data,
                                                                        lambda_reg=self.lambda_reg)

                    if self._zero_centered_grad_penalty:
                        loss -= self._ganloss.calc_zero_centered_GP(self._discriminator,
                                                                    x_gen.data,
                                                                    y_batch.data,
                                                                    cond_batch.data)

                    d_optimizer.zero_grad()
                    loss.backward()
                    d_optimizer.step()
                dis_epoch_loss.append(loss.item())

                for _ in range(self._iters_generator):
                    y_gen = self.generate(cond_batch, normalise=False)
                    if self._instance_noise_std:
                        y_batch = self.instance_noise(y_batch, self._instance_noise_std)
                    if self._task == "CRAMER":
                        y_gen_prime = self.generate(condition=cond_batch, normalise=False)
                        loss = self._ganloss.g_loss(self.loss(y_gen, cond_batch),
                                                    self.loss(y_gen_prime, cond_batch),
                                                    self.loss(y_batch, cond_batch))
                    else:
                        loss = self._ganloss.g_loss(self.loss(y_gen, cond_batch))
                    g_optimizer.zero_grad()
                    loss.backward()
                    g_optimizer.step()
                gen_epoch_loss.append(loss.item())

            if self.burn_in_period is not None:
                if epoch >= self.burn_in_period:
                    if not self.gen_average_weights:
                        for param in self._generator.parameters():
                            self.gen_average_weights.append(param.detach().clone())
                    else:
                        for av_weight, weight in zip(self.gen_average_weights, self._generator.parameters()):
                            av_weight.data = self.averaging_coeff * av_weight.data + \
                                             (1 - self.averaging_coeff) * weight.data

            if self.logger:
                if self.attention_net:
                    self.logger._experiment.log_metric("GAMMA", self.attention_net.gamma.item(), step=self.logger._epoch)
                self.logger.log_losses([dis_epoch_loss, gen_epoch_loss])
                self.logger.log_validation_metrics(self._y_model, y, condition, self,
                                                   (condition[:, :self._psi_dim].min(dim=0)[0].view(-1),
                                                    condition[:, :self._psi_dim].max(dim=0)[0].view(-1)),
                                                   batch_size=1000, calculate_validation_set=False)
                self.logger.add_up_epoch()

        if self.burn_in_period is not None:
            for av_weight, weight in zip(self.gen_average_weights, self._generator.parameters()):
                weight.data = av_weight.data
        return self

    def generate(self, condition, normalise=True):
        if normalise:
            condition = (condition - self._cond_mean) / self._cond_std
            condition[torch.isnan(condition)] = 0
        n = len(condition)
        z = torch.randn(n, self._noise_dim).to(self.device)
        y = self._generator(z, condition)
        if normalise:
            y = y * self._y_std + self._y_mean
        return y

    def log_density(self, y, condition):
        return None
