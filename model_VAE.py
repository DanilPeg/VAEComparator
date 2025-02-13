# model_VAE.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Гиперпараметры
latent_dim = 128
projection_dim = 64
batch_size = 256       # число сэмплов (одновременно обрабатывается N оригиналов → 2N видов)
epochs = 10
learning_rate = 1e-3
beta = 1.0             # вес KL-дивергенции
lambda_contrast = 1.0  # вес контрастивной потери
temperature = 0.5      # параметр температуры для NT‑Xent loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################################
# Кастомизированный датасет, возвращающий две аугментированные версии изображения
##########################################################################
class CIFAR100Pair(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, download=False, second_transform=None):
        super(CIFAR100Pair, self).__init__(root=root, train=train, transform=transform, download=download)
        # Если вторая трансформация не задана, используем первую
        self.second_transform = second_transform if second_transform is not None else transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        from PIL import Image
        img = Image.fromarray(img)
        # Первая аугментация
        if self.transform is not None:
            img1 = self.transform(img)
        else:
            img1 = img
        # Вторая аугментация
        if self.second_transform is not None:
            img2 = self.second_transform(img)
        else:
            img2 = img
        return img1, img2, target

# Аугментации для контрастивного обучения
contrast_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
])

# Функция для получения DataLoader
def get_dataloader(root='./data', batch_size=256, train=True):
    dataset = CIFAR100Pair(root=root, train=train, download=True, transform=contrast_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    return dataloader

##########################################################################
# Определение архитектуры
##########################################################################
# Остаточный блок
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

# Энкодер
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        # Входное изображение 3x32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)    # (64,32,32)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)   # (128,16,16)
        self.bn2 = nn.BatchNorm2d(128)
        self.resblock = ResidualBlock(128)                                   # (128,16,16)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # (256,8,8)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # (512,4,4)
        self.bn4 = nn.BatchNorm2d(512)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))   # (64,32,32)
        x = F.relu(self.bn2(self.conv2(x)))   # (128,16,16)
        x = self.resblock(x)                  # (128,16,16)
        x = F.relu(self.bn3(self.conv3(x)))   # (256,8,8)
        x = F.relu(self.bn4(self.conv4(x)))   # (512,4,4)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Декодер
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # (256,8,8)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # (128,16,16)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # (64,32,32)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)     # (3,32,32)
    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 512, 4, 4)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.sigmoid(self.deconv4(x))
        return x

# Проекционная голова для контрастивного обучения
class ProjectionHead(nn.Module):
    def __init__(self, latent_dim, projection_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, projection_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# Контрастивный VAE
class ContrastiveVAE(nn.Module):
    def __init__(self, latent_dim, projection_dim):
        super(ContrastiveVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.projection_head = ProjectionHead(latent_dim, projection_dim)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        proj = self.projection_head(mu)  # используем mu для получения детерминированного представления
        return recon_x, mu, logvar, proj
    def get_embedding(self, x):
        # Для инференса возвращаем нормированный эмбеддинг из проекционной головы.
        mu, _ = self.encoder(x)
        proj = self.projection_head(mu)
        return proj

##########################################################################
# Контрастивная (NT-Xent) потеря
##########################################################################
def nt_xent_loss(embeddings, temperature):
    """
    embeddings: тензор размера (2N, d), где для каждого i из [0, N-1] положительная пара — элемент i+N.
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]
    if batch_size % 2 != 0:
        raise ValueError("Размер батча должен быть чётным.")
    N = batch_size // 2
    # Косинусное сходство между всеми парами
    sim_matrix = torch.matmul(embeddings, embeddings.T)  # (2N, 2N)
    sim_matrix = sim_matrix / temperature
    # Замещаем диагональные элементы (самосходство) очень малыми значениями
    mask = torch.eye(batch_size, device=device).bool()
    sim_matrix.masked_fill_(mask, -9e15)
    # Для каждого примера положительная пара: для i, если i < N, то положительный j = i+N, иначе j = i-N.
    labels = torch.arange(N, device=device)
    labels = torch.cat([labels + N, labels])
    exp_sim = torch.exp(sim_matrix)
    denom = exp_sim.sum(dim=1)
    pos_sim = torch.exp(sim_matrix[torch.arange(batch_size), labels])
    loss = -torch.log(pos_sim / denom)
    loss = loss.mean()
    return loss

##########################################################################
# Функция потерь для контрастивного VAE
##########################################################################
def loss_function(recon_x, x, mu, logvar, proj, temperature, lambda_contrast, beta):
    # Потеря реконструкции (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL-дивергенция
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Контрастивная потеря
    contrast_loss = nt_xent_loss(proj, temperature)
    total_loss = recon_loss + beta * kld + lambda_contrast * contrast_loss
    return total_loss, recon_loss, kld, contrast_loss

##########################################################################
# Тренировочный цикл
##########################################################################
def train(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss_epoch = 0
    for batch_idx, (img1, img2, _) in enumerate(train_loader):
        # img1 и img2 – две аугментированные версии одного изображения (форма: (N, C, H, W))
        img1 = img1.to(device)
        img2 = img2.to(device)
        # Объединяем батчи: (2N, C, H, W)
        x = torch.cat([img1, img2], dim=0)
        optimizer.zero_grad()
        recon_x, mu, logvar, proj = model(x)
        loss, recon_loss, kld, contrast_loss = loss_function(recon_x, x, mu, logvar, proj, temperature, lambda_contrast, beta)
        loss.backward()
        optimizer.step()
        total_loss_epoch += loss.item()
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss per sample: {loss.item()/x.size(0):.4f} | Recon: {recon_loss.item()/x.size(0):.4f}, KL: {kld.item()/x.size(0):.4f}, Contrast: {contrast_loss.item():.4f}")
    avg_loss = total_loss_epoch / (len(train_loader.dataset)*2)  # делим на общее число примеров (2 вида)
    print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

def main():
    dataloader = get_dataloader(batch_size=batch_size, train=True)
    model = ContrastiveVAE(latent_dim, projection_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(1, epochs + 1):
        train(model, dataloader, optimizer, device, epoch)
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/contrastive_vae_cifar100.pth")
    print("Model saved to models/contrastive_vae_cifar100.pth")

if __name__ == "__main__":
    main()
