# Autoencoding beyond pixels using a learned similarity metric
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv

_use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if _use_cuda else 'cpu')

_transformer = tv.transforms.Compose([
    tv.transforms.Resize([64, 64]),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5] * 3, [0.5] * 3)
])

train_iter = torch.utils.data.DataLoader(
    dataset=tv.datasets.LSUN(
        root='../../Datasets/LSUN/',
        transform=_transformer,
        classes=['bedroom_train']
    ),
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)


class Encoder(nn.Module):
    def __init__(self, in_c, z_dim=64):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.features = nn.Sequential(
            # nc x 64 x 64
            nn.Conv2d(in_c, 64, 5, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5, 2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),  # nf*4 x 8 x 8
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, z_dim * 2)
        )

    def forward(self, x):
        hidden = self.features(x)
        common = self.fc(hidden.view(x.size(0), -1))
        return common[:, :self.z_dim], common[:, self.z_dim:]


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.ReLU()
        )
        self.dconv = nn.Sequential(
            # 256 x 8 x 8
            nn.ConvTranspose2d(256, 256, 5, 2, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, 2, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 32, 5, 2, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 5, 1, 2),
            nn.Tanh()
        )

    def forward(self, z):
        fc_z = self.fc(z).view(-1, 256, 8, 8)
        return self.dconv(fc_z)


class VAE(nn.Module):
    def __init__(self, in_c, z_dim):
        super(VAE, self).__init__()
        self.enc = Encoder(in_c, z_dim)
        self.dec = Decoder(z_dim)
        self.to(DEVICE)

    def kld_loss(self, mu, logvar):
        return 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(1).mean()

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn(mu.size()).to(DEVICE)
        rec_x = self.dec(z)
        return self.kld_loss(mu, logvar), z, rec_x


class Discriminator(nn.Module):
    def __init__(self, in_c):
        super(Discriminator, self).__init__()
        self.common = nn.Sequential(
            nn.Conv2d(in_c, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.lth = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU()
        )
        self.scale_out = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.to(DEVICE)

    def forward(self, x):
        features = self.common(x)
        lth = self.lth(features.view(features.size(0), -1))
        return self.scale_out(lth)

    def lth_features(self, x):
        features = self.common(x)
        return self.lth(features.view(features.size(0), -1))


in_c = 3
z_dim = 64
lr = 3e-4
gamma = 1e-1
n_epochs = 100
save_dir = './vae_gan/'
os.makedirs(save_dir, exist_ok=True)

vae = VAE(in_c, z_dim)
dis = Discriminator(in_c)

enc_trainer = optim.RMSprop(vae.enc.parameters(), lr)
dec_trainer = optim.RMSprop(vae.dec.parameters(), lr)
dis_trainer = optim.RMSprop(dis.parameters(), lr * 0.1)
rec_criterion = nn.MSELoss()
bce_criterion = nn.BCELoss()

for epoch in range(n_epochs):
    vae.train()
    dis.train()
    for batch_idx, (x, _) in enumerate(train_iter):
        x = x.to(DEVICE)
        kld_loss, qz, rx = vae(x)
        rec_loss = rec_criterion(rx, x)
        pz = torch.randn(x.size(0), z_dim).to(DEVICE)
        px = vae.dec(pz)
        # gan_loss = (torch.log(dis(x) + EPS) + torch.log(1 - dis(rx) + EPS) + torch.log(1 - dis(px) + EPS)).sum(1).mean()
        # 用下面的loss要数值更加稳定点
        rl_score, fk_score, rc_score = dis(x), dis(px), dis(rx)
        gan_loss = bce_criterion(rl_score, torch.ones_like(rl_score)) + bce_criterion(fk_score, torch.zeros_like(
            fk_score)) + bce_criterion(rc_score, torch.zeros_like(rc_score))

        # train enc
        loss_enc = kld_loss + rec_loss
        enc_trainer.zero_grad()
        loss_enc.backward(retain_graph=True)
        enc_trainer.step()
        # train dec
        loss_dec = gamma * rec_loss - gan_loss
        dec_trainer.zero_grad()
        loss_dec.backward(retain_graph=True)
        dec_trainer.step()
        # train dis
        loss_dis = gan_loss
        dis_trainer.zero_grad()
        loss_dis.backward()
        dis_trainer.step()

        if batch_idx % 50 == 0:
            print('[%2d/%2d] [%5d/%5d] ' % (epoch, n_epochs, batch_idx, len(train_iter)) +
                  'kld_loss: %0.3f ' % kld_loss.item() +
                  'rec_loss: %0.3f ' % rec_loss.item() +
                  'gan_loss: %0.3f ' % gan_loss.item() +
                  'loss_enc: %0.3f ' % loss_enc.item() +
                  'loss_dec: %0.3f ' % loss_dec.item() +
                  'loss_dis: %0.3f ' % loss_dis.item() +
                  'FK-score/RL-score/RC-score: [%0.3f/%0.3f/%0.3f]' %
                  (fk_score.mean().item(), rl_score.mean().item(), rc_score.mean().item()))

            tv.utils.save_image(rx.detach()[:64] * 0.5 + 0.5, save_dir + 'rx_{}_{}.png'.format(epoch, batch_idx))
            tv.utils.save_image(px.detach()[:64] * 0.5 + 0.5, save_dir + 'px_{}_{}.png'.format(epoch, batch_idx))
