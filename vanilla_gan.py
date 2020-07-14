import os

import torch as T
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torch.autograd import Variable
from torch.optim import lr_scheduler

lr = 1e-3
epoch = 0
n_epochs = 100

save_dir = './results_vanilla_gan_1/'
os.makedirs(save_dir, exist_ok=True)

print_every = 100
epoch_lr_decay = 50
save_epoch_freq = 1

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
    nn.Linear(dim_z, 256),
    nn.ReLU(),
    nn.Linear(256, dim_im),
    nn.ReLU()
).cuda()

D = T.nn.Sequential(
    nn.Linear(dim_im, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
).cuda()

opt_G = optim.Adam(G.parameters(), lr, betas=[0.5, 0.99])
opt_D = optim.Adam(D.parameters(), lr, betas=[0.5, 0.99])
scheduler_G = lr_scheduler.StepLR(opt_G, step_size=2, gamma=0.8)
scheduler_D = lr_scheduler.StepLR(opt_D, step_size=2, gamma=0.8)

G.train()
D.train()

# resume
if epoch >= 1:
    ckpt = T.load(save_dir + 'ckpt_{}.ptz'.format(epoch))
    lr = ckpt['lr']
    epoch = ckpt['epoch']
    G.load_state_dict(ckpt['G'])
    D.load_state_dict(ckpt['D'])


def gaussian(size, mean=0, std=1):
    return T.normal(T.ones(size) * mean, std)


# train
for _ in range(epoch, n_epochs):
    epoch += 1
    if epoch > epoch_lr_decay:
        scheduler_G.step()
        scheduler_D.step()

    _batch = 0
    for X, _ in train_iter:
        _batch += 1
        # G
        x_real = Variable(X).view(-1, dim_im).cuda()
        z = Variable(gaussian([batch_size, dim_z])).cuda()
        fake_x = G(z)
        fake_score = D(fake_x)

        # loss_G = T.mean(T.log(T.ones_like(fake_score) - fake_score))
        loss_G = -T.mean(T.log(fake_score))  # 相比较上面的loss， 这个收敛的更快

        G.zero_grad()
        loss_G.backward()
        opt_G.step()

        # D
        fake_score = D(fake_x.detach())
        real_score = D(x_real)

        loss_D = - T.mean(T.log(T.ones_like(fake_score) - fake_score) +
                          T.log(real_score))

        D.zero_grad()
        loss_D.backward()
        opt_D.step()

        if _batch % print_every == 0:
            print('Epoch %d Batch %d ' % (epoch, _batch) +
                  'Loss D: %0.3f ' % loss_D.data[0] +
                  'Loss G: %0.3f ' % loss_G.data[0] +
                  'F-score/R-score: [ %0.3f / %0.3f ]' %
                  (T.mean(fake_score.data), T.mean(real_score.data)))

            _imags = fake_x.view(batch_size, 1, 28, 28).data
            tv.utils.save_image(_imags, save_dir + '{}_{}.png'.format(epoch, _batch))

    if epoch % save_epoch_freq == 0:
        T.save({
            'lr': lr,
            'epoch': epoch,
            'G': G.state_dict(),
            'D': D.state_dict()
        }, save_dir + 'ckpt_{}.ptz'.format(epoch))
