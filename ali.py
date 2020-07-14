import os

import torch as T
import torch.nn as nn
import torch.optim as optim
import torchvision as tv

from utils import plot_q_z

_use_cuda = T.cuda.is_available()
DEVICE = T.device('cuda' if _use_cuda else 'cpu')

save_dir = './ali/'
os.makedirs(save_dir, exist_ok=True)

print_every = 100
save_epoch_freq = 2

dim_z = 10
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

P = nn.Sequential(
    nn.Linear(dim_z, 128),
    nn.ReLU(),
    nn.Linear(128, dim_im),
    nn.Sigmoid()
).to(DEVICE)

Q = nn.Sequential(
    nn.Linear(dim_im, 128),
    nn.ReLU(),
    nn.Linear(128, dim_z)
).to(DEVICE)

D = T.nn.Sequential(
    nn.Linear(dim_im + dim_z, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
).to(DEVICE)


def train(data_iter, n_epochs, lr):
    opt_g = optim.Adam(list(P.parameters()) + list(Q.parameters()), lr, [0.5, 0.99])
    opt_d = optim.Adam(D.parameters(), lr, [0.5, 0.99])

    P.train()
    Q.train()
    D.train()

    for epoch in range(n_epochs):

        _batch = 0
        for X, l in data_iter:
            _batch += 1

            x = X.view(-1, dim_im).to(DEVICE)
            z = T.randn(batch_size, dim_z).to(DEVICE)
            px = P(z)
            pxz = T.cat([px, z], 1)
            qz = Q(x)
            xqz = T.cat([x, qz], 1)

            pxz_score = D(pxz.detach())
            xqz_score = D(xqz.detach())

            # train D
            loss_d = -T.mean(T.log(xqz_score + 1e-10) + T.log(1 - pxz_score + 1e-10))

            D.zero_grad()
            loss_d.backward()
            opt_d.step()

            # train G
            pxz_score = D(pxz)
            xqz_score = D(xqz)
            loss_g = -T.mean(T.log(pxz_score + 1e-10) + T.log(1 - xqz_score + 1e-10))

            P.zero_grad()
            Q.zero_grad()
            loss_g.backward()
            opt_g.step()

            if _batch % print_every == 0:
                print('Epoch %d Batch %d ' % (epoch, _batch) +
                      'loss_d: %0.3f ' % loss_d.item() +
                      'loss_g: %0.3f ' % loss_g.item() +
                      'pxz_score: %0.3f ' % T.mean(pxz_score).item() +
                      'xqz_score: %0.3f ' % T.mean(xqz_score).item())

                _imags = px.view(batch_size, 1, 28, 28).data
                tv.utils.save_image(_imags, save_dir + '{}_{}.png'.format(epoch, _batch))

                # test z
                with T.no_grad():
                    x, l = next(iter(test_iter))
                    x = x.view(-1, dim_im).to(DEVICE)
                    qz = Q(x)
                    plot_q_z(qz.detach().cpu().numpy(), l, save_dir + 'z_{}_{}.png'.format(epoch, _batch))


if __name__ == '__main__':
    train(train_iter, 20, 1e-3)
