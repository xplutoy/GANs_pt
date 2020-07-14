import os

import torch as T
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torch.optim import lr_scheduler

from utils import print_network, add_noise

_use_cuda = T.cuda.is_available()
DEVICE = T.device('cuda' if _use_cuda else 'cpu')

lr = 2e-4
n_epochs = 30

instance_noise_trick = False
initial_noise_strength = 0.1
anneal_epoch = int(n_epochs * 2 / 3)

save_dir = './results_dcgan/'
os.makedirs(save_dir, exist_ok=True)

print_every = 50
save_epoch_freq = 5

nz = 100
nc = 3
ndf = 128
ngf = 128
im_size = [64, 64]
batch_size = 128

_transformer = tv.transforms.Compose([
    tv.transforms.Resize(im_size),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5] * 3, [0.5] * 3)
])

train_iter = T.utils.data.DataLoader(
    dataset=tv.datasets.LSUN(
        root='../../Datasets/LSUN/',
        transform=_transformer,
        classes=['bedroom_train']
    ),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)


class generator(nn.Module):
    """
    dc_gan for lsun datasets
    """

    def __init__(self, nz, nc, ngf):
        super(generator, self).__init__()
        self.net = nn.Sequential(
            # layer 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            # layer 2 (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # layer 3
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # layer 4
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # layer 5
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.to(DEVICE)
        print_network(self)

    def forward(self, z):
        return nn.parallel.data_parallel(self.net, z)


class discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(discriminator, self).__init__()
        self.net = nn.Sequential(
            # layer 1
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # layer 2
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # layer 3
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            # layer 3
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # layer 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.to(DEVICE)
        print_network(self)

    def forward(self, x):
        return nn.parallel.data_parallel(self.net, x).view(-1, 1).squeeze(1)


G = generator(nz, nc, ngf)
D = discriminator(3, ndf)
opt_G = optim.Adam(G.parameters(), lr, betas=[0.5, 0.999])
opt_D = optim.Adam(D.parameters(), lr, betas=[0.5, 0.999])
scheduler_lr = lr_scheduler.StepLR(opt_G, step_size=1, gamma=0.9)
criterion = nn.BCELoss()

for epoch in range(0, n_epochs):
    G.train()
    D.train()
    _batch = 0
    scheduler_lr.step()
    for X, _ in train_iter:
        _batch += 1

        real_x = X.to(DEVICE)
        z = T.randn(real_x.size(0), nz, 1, 1, device=DEVICE)
        fake_x = G(z)

        # instance noise trick
        if instance_noise_trick:
            real_x = add_noise(real_x, initial_noise_strength, anneal_epoch, epoch)
            fake_x = add_noise(fake_x, initial_noise_strength, anneal_epoch, epoch)

        fake_score = D(fake_x.detach())
        real_score = D(real_x)

        D.zero_grad()
        lss_D = criterion(real_score, T.ones_like(real_score)) + \
                criterion(fake_score, T.zeros_like(fake_score))
        lss_D.backward()
        opt_D.step()

        fake_score = D(fake_x)
        real_score = D(real_x)

        G.zero_grad()
        lss_G = criterion(fake_score, T.ones_like(fake_score))
        lss_G.backward()
        opt_G.step()

        if _batch % print_every == 0:
            print('[%2d/%2d] [%5d/%5d] ' % (epoch, n_epochs, _batch, len(train_iter)) +
                  'loss_D: %0.3f ' % lss_D.item() +
                  'loss_G: %0.3f ' % lss_G.item() +
                  'F-score/R-score: [%0.3f/%0.3f]' %
                  (T.mean(fake_score).item(), T.mean(real_score).item()))

            tv.utils.save_image(fake_x.detach()[:64] * 0.5 + 0.5, save_dir + '{}_{}.png'.format(epoch + 1, _batch))

    if (epoch + 1) % save_epoch_freq == 0:
        T.save(D.state_dict(), 'dcgan_netd_{}.pth'.format(epoch + 1))
        T.save(G.state_dict(), 'dcgan_netg_{}.pth'.format(epoch + 1))
