import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv

from utils import one_hot, DEVICE

save_dir = './cgan/'
os.makedirs(save_dir, exist_ok=True)

print_every = 100
save_epoch_freq = 2

dim_z = 10
dim_c = 10
dim_im = 784

batch_size = 64

train_iter = torch.utils.data.DataLoader(
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

test_iter = torch.utils.data.DataLoader(
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


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_z + dim_c, 256),
            nn.ReLU(),
            nn.Linear(256, dim_im),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.net(input)


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_im + dim_c, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.net(input)


def train(G, D, data_iter, n_epochs, lr):
    opt_g = optim.Adam(G.parameters(), lr, betas=[.5, .999])
    opt_d = optim.Adam(D.parameters(), lr, betas=[.5, .999])

    G.train()
    D.train()

    for epoch in range(n_epochs):
        for i, (X, L) in enumerate(data_iter):
            x_real = X.view(-1, dim_im).to(DEVICE)
            l = one_hot(L, 10).to(DEVICE)
            z = torch.randn(l.size(0), dim_z, device=DEVICE)

            fake_x = G(torch.cat([z, l], 1))

            fake_score = D(torch.cat([fake_x.detach(), l], 1))
            real_score = D(torch.cat([x_real, l], 1))

            loss_d = -torch.mean(torch.log(real_score + 1e-10) + torch.log(1 - fake_score + 1e-10))
            D.zero_grad()
            loss_d.backward()
            opt_d.step()

            fake_score = D(torch.cat([fake_x, l], 1))
            real_score = D(torch.cat([x_real, l], 1))

            loss_g = torch.mean(torch.log(1 - fake_score + 1e-10))
            # loss_g = -torch.mean(torch.log(fake_score + 1e-10))
            G.zero_grad()
            loss_g.backward()
            opt_g.step()

            if (i + 1) % print_every == 0:
                print('Epoch %d Batch %d ' % (epoch + 1, i + 1) +
                      'Loss D: %0.3f ' % loss_d.item() +
                      'Loss G: %0.3f ' % loss_g.item() +
                      'fake_score: %0.3f ' % torch.mean(fake_score).item() +
                      'real_score: %0.3f ' % torch.mean(real_score).item())

                _imags = fake_x.view(batch_size, 1, 28, 28).data
                tv.utils.save_image(_imags, save_dir + '{}_{}.png'.format(epoch + 1, i + 1))
        if (epoch + 1) % save_epoch_freq == 0:
            torch.save(G.state_dict(), save_dir + 'net_g.pt')
            torch.save(G.state_dict(), save_dir + 'net_d.pt')
        # test
        with torch.no_grad():
            G.eval()
            z = torch.randn(1, dim_z, device=DEVICE).repeat(10, 1)
            c = one_hot(torch.range(0, 9, dtype=torch.long), 10).to(DEVICE)
            fake_x = G(torch.cat([z, c], 1))
            _imags = fake_x.view(-1, 1, 28, 28)
            tv.utils.save_image(_imags, save_dir + '{}.png'.format(epoch + 1))


if __name__ == '__main__':
    d, g = D().to(DEVICE), G().to(DEVICE)
    train(g, d, train_iter, 20, 1e-3)
