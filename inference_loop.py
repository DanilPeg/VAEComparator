# inference_loop.py
import sys
import torch
from PIL import Image
import torchvision.transforms as transforms
from model_AE import ResidualAutoEncoder, get_embedding

def load_image(image_path):
    """
    Загружает изображение, преобразует в RGB, изменяет размер до 32x32 и конвертирует в тензор.
    """
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    return image_tensor

def main(image_path1, image_path2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualAutoEncoder().to(device)
    try:
        model.load_state_dict(torch.load("ae_model.pth", map_location=device))
    except Exception as e:
        print("Ошибка загрузки модели. Убедитесь, что модель обучена и файл ae_model.pth существует.")
        return
    model.eval()
    
    image_tensor1 = load_image(image_path1).to(device)
    image_tensor2 = load_image(image_path2).to(device)
    
    embedding1 = get_embedding(model, image_tensor1)
    embedding2 = get_embedding(model, image_tensor2)
    
    print("Эмбеддинг 1:", embedding1.cpu().numpy())
    print("Эмбеддинг 2:", embedding2.cpu().numpy())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Использование: python inference_loop.py <image1> <image2>")
    else:
        main(sys.argv[1], sys.argv[2])
