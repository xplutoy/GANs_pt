import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv

from utils import *

lr = 2e-4
n_epochs = 20
batch_size = 200

save_dir = './ssl_gan/'
os.makedirs(save_dir, exist_ok=True)

print_every = 5
save_epoch_freq = 5

img_shape = [32, 32]

train_iter = torch.utils.data.DataLoader(
    dataset=tv.datasets.CIFAR10(
        root='../../Datasets/CIFAR10/',
        transform=tv.transforms.Compose([
            tv.transforms.Resize(img_shape),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.5] * 3, [0.5] * 3)]),
        train=True,
        download=True
    ),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)


def label_data_batch(bs, max_size=200):
    X = tv.datasets.CIFAR10(
        root='../../Datasets/CIFAR10/',
        transform=tv.transforms.Compose([
            tv.transforms.Resize(img_shape),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        train=False,
        download=False
    )
    L = len(X)
    assert bs < L and max_size <= L
    batch = [X[i] for i in np.random.choice(max_size, bs)]
    batch_X, batch_L = list(zip(*batch))
    batch_X, batch_L = torch.stack(batch_X), torch.from_numpy(np.array(batch_L))
    return batch_X, batch_L


# https://github.com/eli5168/improved_gan_pytorch/blob/master/improved_GAN.py
class _netG(nn.Module):
    def __init__(self, nz=100, nc=3, ngf=32):
        super(_netG, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.to(DEVICE)
        print_network(self)

    def forward(self, z):
        return self.net(z)


class _netD(nn.Module):
    def __init__(self, nc=3, ndf=32):
        super(_netD, self).__init__()
        self.conv = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
            # output is ndf * 8 * 2 * 2
        )
        self.fc = nn.Linear(1024, 10)
        self.to(DEVICE)
        print_network(self)

    def forward(self, x):
        conv = self.conv(x)
        feature = conv.view(-1, 1024)
        logits = self.fc(feature)
        return logits, feature


nc = 3
nz = 100
ngf = 32
ndf = 32

G = _netG(nz, nc, ngf)
D = _netD(nc, ndf)

opt_G = optim.Adam(G.parameters(), lr, betas=[0.5, 0.99])
opt_D = optim.Adam(D.parameters(), lr, betas=[0.5, 0.99])

cl_criterion = nn.CrossEntropyLoss()
fm_criterion = nn.MSELoss()

# 用cifar10的测试机当着有标签的数据， 训练集为不带标签的数据
for epoch in range(n_epochs):
    G.train()
    D.train()

    _batch = 0
    for unl_x, _ in train_iter:
        _batch += 1
        unl_x = unl_x.to(DEVICE)
        x, y = label_data_batch(batch_size)
        x, y = x.to(DEVICE), y.to(DEVICE)

        # train D
        z = torch.randn(x.size(0), nz, 1, 1, device=DEVICE)
        fake_x = G(z)

        lab_logit = D(x)[0]
        unl_logit = D(unl_x)[0]
        fak_logit = D(fake_x.detach())[0]

        logz_lab, logz_unl, logz_fak = log_sum_exp(lab_logit), log_sum_exp(unl_logit), log_sum_exp(fak_logit)

        real_score = torch.mean(torch.exp(logz_unl - F.softplus(logz_unl)))
        fake_score = torch.mean(torch.exp(logz_fak - F.softplus(logz_fak)))

        d_supervised_loss = cl_criterion(lab_logit, y)
        d_unsupervised_loss = - torch.mean(logz_unl - F.softplus(logz_unl) - F.softplus(logz_fak))
        d_loss = d_supervised_loss + d_unsupervised_loss

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # train G
        # # 一般的adv_loss
        # fak_logit = D(fake_x)[-1]
        # logz_fak = log_sum_exp(fak_logit)
        # g_loss = -torch.mean(F.softplus(logz_fak))

        # feature match
        fake_feature = D(fake_x)[1]
        real_feature = D(unl_x)[1]
        g_loss = fm_criterion(fake_feature.mean(0), real_feature.detach().mean(0))
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        if _batch % print_every == 0:
            acc = get_cls_accuracy(lab_logit, y)
            print('[%d/%d] [%d] g_loss: %.3f d_loss: %.3f real_score: %.3f fake_score: %.3f acc: %.3f' % (
                epoch + 1, n_epochs, _batch, g_loss.item(), d_loss.item(), real_score.item(), fake_score.item(),
                acc.item()))
            tv.utils.save_image(fake_x[:16] * 0.5 + 0.5, save_dir + '{}_{}.png'.format(epoch + 1, _batch))

    # 查看对无标签数据的分类准确率
    total_acc = 0
    for x, label in train_iter:
        with torch.no_grad():
            D.eval()
            preds = D(x.to(DEVICE))[0]
            total_acc += get_cls_accuracy(preds, label.to(DEVICE))
    total_acc = total_acc / len(train_iter)
    print('半监督的gan分类器的分类正确率：{}'.format(total_acc))

# 相同结构的一个baseline model 用于比较
baseline = _netD(nc, ndf)
opt_b = optim.Adam(baseline.parameters(), lr, betas=[0.5, 0.99])

for epoch in range(n_epochs):
    baseline.train()
    _batch = 0
    for _ in train_iter:
        _batch += 1
        x, y = label_data_batch(batch_size)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        y_ = baseline(x)[0]
        loss_b = cl_criterion(y_, y)

        opt_b.zero_grad()
        loss_b.backward()
        opt_b.step()

        if _batch % print_every == 0:
            acc = get_cls_accuracy(y_, y)
            print('[%d/%d] [%d] loss: %.3f acc: %.3f' % (epoch + 1, n_epochs, _batch, loss_b.item(), acc.item()))

    # 查看对无标签数据的分类准确率
    total_acc = 0
    for x, label in train_iter:
        with torch.no_grad():
            baseline.eval()
            preds = baseline(x.to(DEVICE))[0]
            total_acc += get_cls_accuracy(preds, label.to(DEVICE))
    total_acc = total_acc / len(train_iter)
    print('Baseline分类正确率：{}'.format(total_acc))
