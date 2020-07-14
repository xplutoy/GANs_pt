import os

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv

_use_cuda = T.cuda.is_available()
DEVICE = T.device('cuda' if _use_cuda else 'cpu')

save_dir = './energy_gan/'
os.makedirs(save_dir, exist_ok=True)

print_every = 100
save_epoch_freq = 2

dim_z = 16
dim_im = 784

batch_size = 64

train_iter = T.utils.data.DataLoader(
    dataset=tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=True,
        download=True
    ),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)

test_iter = T.utils.data.DataLoader(
    dataset=tv.datasets.MNIST(
        root='../../Datasets/MNIST/',
        transform=tv.transforms.ToTensor(),
        train=False,
        download=True
    ),
    batch_size=1000,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)

G = T.nn.Sequential(
    nn.Linear(dim_z, 128),
    nn.ReLU(),
    nn.Linear(128, dim_im),
    nn.Sigmoid()
).to(DEVICE)

D = T.nn.Sequential(
    nn.Linear(dim_im, 32),
    nn.ReLU(),
    nn.Linear(32, dim_im),
).to(DEVICE)


def gaussian(size, mean=0, std=1):
    return T.normal(T.ones(size) * mean, std)


def train(G, D, data_iter, n_epochs, lr, beta):
    opt_g = optim.Adam(G.parameters(), lr, [0.5, 0.99])
    opt_d = optim.Adam(D.parameters(), lr, [0.5, 0.99])

    G.train()
    D.train()

    for epoch in range(n_epochs):

        _batch = 0
        for X, _ in data_iter:
            _batch += 1
            # G
            x_real = X.view(-1, dim_im).to(DEVICE)
            z = gaussian([batch_size, dim_z]).to(DEVICE)
            fake_x = G(z)
            fake_rec = D(fake_x)

            loss_G = F.binary_cross_entropy_with_logits(fake_rec, fake_x)

            G.zero_grad()
            loss_G.backward()
            opt_g.step()

            # D
            fake_x = fake_x.detach()
            fake_rec = D(fake_x)
            real_rec = D(x_real)

            loss_D_1 = F.binary_cross_entropy_with_logits(fake_rec, fake_x)
            loss_D_2 = F.binary_cross_entropy_with_logits(real_rec, x_real)
            loss_D = loss_D_2 + F.relu(beta - loss_D_1)

            D.zero_grad()
            loss_D.backward()
            opt_d.step()

            if _batch % print_every == 0:
                print('Epoch %d Batch %d ' % (epoch, _batch) +
                      'Loss D: %0.3f ' % loss_D.item() +
                      'Loss G: %0.3f ' % loss_G.item() +
                      'Loss D1: %0.3f ' % loss_D_1.item() +
                      'Loss D2: %0.3f ' % loss_D_2.item())

                _imags = fake_x.view(batch_size, 1, 28, 28).data
                tv.utils.save_image(_imags, save_dir + '{}_{}.png'.format(epoch, _batch))


if __name__ == '__main__':
    # beta参数对batch_size敏感
    # batch_size=64时， beta=1.6是OK的
    train(G, D, train_iter, 20, 1e-3, 1.6)
