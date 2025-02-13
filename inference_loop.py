# inference_loop.py
import sys
import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from model_VAE import ContrastiveVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128
projection_dim = 64
model = ContrastiveVAE(latent_dim, projection_dim).to(device)

model_path = os.path.join("models", "contrastive_vae_cifar100.pth")
if not os.path.exists(model_path):
    print("Model not found. Train the model first using model_VAE.py")
    sys.exit(1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((32, 32))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image.to(device)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference_loop.py image1_path image2_path")
        sys.exit(1)
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    img1 = preprocess_image(image1_path)
    img2 = preprocess_image(image2_path)
    with torch.no_grad():
        emb1 = model.get_embedding(img1).cpu().numpy().flatten()
        emb2 = model.get_embedding(img2).cpu().numpy().flatten()
    print("Embedding for image 1:")
    print(emb1)
    print("Embedding for image 2:")
    print(emb2)
    # Вычисление метрик сходства
    euclidean = np.linalg.norm(emb1 - emb2)
    cosine_similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10)
    manhattan = np.sum(np.abs(emb1 - emb2))
    print(f"Euclidean distance: {euclidean:.4f}")
    print(f"Cosine similarity: {cosine_similarity:.4f}")
    print(f"Manhattan distance: {manhattan:.4f}")
