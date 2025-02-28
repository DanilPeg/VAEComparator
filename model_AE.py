# model_AE.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class ResidualBlock(nn.Module):
    """
    Остаточный блок с двумя сверточными слоями, Batch Normalization и ReLU.
    При необходимости используется downsample для согласования размерностей.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class UpBlock(nn.Module):
    """
    Блок для апсемплинга, использующий ConvTranspose2d для увеличения разрешения.
    """
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, 
                               padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.up(x)

class ResidualAutoEncoder(nn.Module):
    """
    Усовершенствованный автоэнкодер с остаточными блоками.
    Энкодер сворачивает изображение (3x32x32) до компактного эмбеддинга,
    а декодер восстанавливает изображение из эмбеддинга.
    """
    def __init__(self, embedding_dim=256):
        super(ResidualAutoEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Энкодер
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)   # 32x32
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)    # 16x16
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)   # 8x8
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)   # 4x4
        
        self.fc_enc = nn.Linear(512 * 4 * 4, embedding_dim)
        
        # Декодер
        self.fc_dec = nn.Linear(embedding_dim, 512 * 4 * 4)
        self.up1 = UpBlock(512, 256)  # 4x4 -> 8x8
        self.up2 = UpBlock(256, 128)  # 8x8 -> 16x16
        self.up3 = UpBlock(128, 64)   # 16x16 -> 32x32
        
        self.refine = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # значения в диапазоне [0, 1]
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def encode(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc_enc(x)
        return embedding
    
    def decode(self, embedding):
        x = self.fc_dec(embedding)
        x = x.view(-1, 512, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.refine(x)
        reconstruction = self.final_conv(x)
        return reconstruction
    
    def forward(self, x):
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction, embedding

def train_ae(num_epochs=50, batch_size=128, learning_rate=1e-3, embedding_dim=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = ResidualAutoEncoder(embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            reconstructions, _ = model(images)
            loss = criterion(reconstructions, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
        epoch_loss /= len(train_loader.dataset)
        print(f"Эпоха [{epoch+1}/{num_epochs}], Потеря: {epoch_loss:.4f}")
    
    torch.save(model.state_dict(), "ae_model.pth")
    print("Модель сохранена как ae_model.pth")

def get_embedding(model, image_tensor):
    """
    Получение эмбеддинга для одного изображения.
    image_tensor: тензор изображения размером (C, H, W)
    Возвращает: эмбеддинг размерности (embedding_dim)
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0)
        _, embedding = model(image_tensor)
    return embedding.squeeze(0)

if __name__ == "__main__":
    train_ae(num_epochs=50)
