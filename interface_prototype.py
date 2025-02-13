# interface_prototype.py
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QFrame)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from model_VAE import ContrastiveVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128
projection_dim = 64

class СравнениеИзображений(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Сравнение изображений")
        self.setGeometry(100, 100, 1000, 800)
        self.image1 = None
        self.image2 = None
        self.image1_path = None
        self.image2_path = None
        self.init_model()
        self.init_ui()

    def init_model(self):
        self.model = ContrastiveVAE(latent_dim, projection_dim).to(device)
        model_path = "models/contrastive_vae_cifar100.pth"
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print("Модель загружена.")
        except Exception as e:
            print("Ошибка при загрузке модели:", e)
        self.model.eval()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Заголовок
        title_label = QLabel("Сравнение изображений")
        title_font = QFont("Arial", 24, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Блок для отображения изображений
        img_layout = QHBoxLayout()
        # Изображение 1
        vbox1 = QVBoxLayout()
        self.img1_label = QLabel("Изображение 1")
        self.img1_label.setAlignment(Qt.AlignCenter)
        self.img1_display = QLabel()
        self.img1_display.setFixedSize(300, 300)
        self.img1_display.setStyleSheet("border: 2px solid #000;")
        vbox1.addWidget(self.img1_label)
        vbox1.addWidget(self.img1_display)
        # Изображение 2
        vbox2 = QVBoxLayout()
        self.img2_label = QLabel("Изображение 2")
        self.img2_label.setAlignment(Qt.AlignCenter)
        self.img2_display = QLabel()
        self.img2_display.setFixedSize(300, 300)
        self.img2_display.setStyleSheet("border: 2px solid #000;")
        vbox2.addWidget(self.img2_label)
        vbox2.addWidget(self.img2_display)
        img_layout.addLayout(vbox1)
        img_layout.addLayout(vbox2)
        main_layout.addLayout(img_layout)

        # Кнопки загрузки изображений
        btn_layout = QHBoxLayout()
        load_btn1 = QPushButton("Загрузить изображение 1")
        load_btn1.setFont(QFont("Arial", 14))
        load_btn1.clicked.connect(self.load_image1)
        load_btn2 = QPushButton("Загрузить изображение 2")
        load_btn2.setFont(QFont("Arial", 14))
        load_btn2.clicked.connect(self.load_image2)
        btn_layout.addWidget(load_btn1)
        btn_layout.addWidget(load_btn2)
        main_layout.addLayout(btn_layout)

        # Кнопка для вычисления сходства
        compute_btn = QPushButton("Вычислить сходство")
        compute_btn.setFont(QFont("Arial", 16, QFont.Bold))
        compute_btn.clicked.connect(self.compute_similarity)
        main_layout.addWidget(compute_btn)

        # Отображение результатов
        self.results_label = QLabel("")
        self.results_label.setFont(QFont("Arial", 16))
        self.results_label.setAlignment(Qt.AlignCenter)
        self.results_label.setWordWrap(True)
        main_layout.addWidget(self.results_label)

        # Легенда для индикаторов сходства с цветными блоками внутри которых размещены надписи
        legend = QLabel()
        legend.setFont(QFont("Arial", 14))
        legend.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        legend.setText(
            "<b>Легенда для индикаторов сходства:</b><br>"
            "<span style='background-color:#00cc00; color:#000; padding:4px 20px; display:inline-block;'>Высокое сходство</span><br>"
            "<span style='background-color:#99ff99; color:#000; padding:4px 20px; display:inline-block;'>Умеренное сходство</span><br>"
            "<span style='background-color:#ffff66; color:#000; padding:4px 20px; display:inline-block;'>Низкое сходство</span><br>"
            "<span style='background-color:#ff6666; color:#000; padding:4px 20px; display:inline-block;'>Нет сходства</span>"
        )
        legend.setAlignment(Qt.AlignLeft)
        main_layout.addWidget(legend)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.setStyleSheet("""
            QWidget {
                background-color: #f8f8f8;
            }
            QPushButton {
                background-color: #007ACC;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005F99;
            }
            QLabel {
                color: #333;
            }
        """)

    def load_image1(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите изображение 1", "", "Изображения (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image1_path = file_name
            self.image1 = Image.open(file_name).convert("RGB")
            pixmap = QPixmap(file_name).scaled(self.img1_display.width(), self.img1_display.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img1_display.setPixmap(pixmap)

    def load_image2(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите изображение 2", "", "Изображения (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image2_path = file_name
            self.image2 = Image.open(file_name).convert("RGB")
            pixmap = QPixmap(file_name).scaled(self.img2_display.width(), self.img2_display.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img2_display.setPixmap(pixmap)

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0).to(device)

    def get_indicator_color(self, metric, value):
        # Пороговые значения для нормализованных эмбеддингов:
        if metric == "euclidean":
            if value < 0.6:
                return "#00cc00"  # Высокое сходство
            elif value < 1.0:
                return "#99ff99"  # Умеренное сходство
            elif value < 1.4:
                return "#ffff66"  # Низкое сходство
            else:
                return "#ff6666"  # Нет сходства
        elif metric == "cosine":
            if value > 0.9:
                return "#00cc00"  # Высокое сходство
            elif value > 0.8:
                return "#99ff99"  # Умеренное сходство
            elif value > 0.7:
                return "#ffff66"  # Низкое сходство
            else:
                return "#ff6666"  # Нет сходства
        elif metric == "manhattan":
            if value < 0.8:
                return "#00cc00"
            elif value < 1.2:
                return "#99ff99"
            elif value < 1.6:
                return "#ffff66"
            else:
                return "#ff6666"
        else:
            return "#cccccc"

    def compute_similarity(self):
        if self.image1 is None or self.image2 is None:
            self.results_label.setText("Пожалуйста, загрузите оба изображения.")
            return
        img1_tensor = self.preprocess_image(self.image1)
        img2_tensor = self.preprocess_image(self.image2)
        with torch.no_grad():
            emb1 = self.model.get_embedding(img1_tensor).cpu().numpy().flatten()
            emb2 = self.model.get_embedding(img2_tensor).cpu().numpy().flatten()
        euclidean = np.linalg.norm(emb1 - emb2)
        cosine_similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10)
        manhattan = np.sum(np.abs(emb1 - emb2))
        color_euc = self.get_indicator_color("euclidean", euclidean)
        color_cos = self.get_indicator_color("cosine", cosine_similarity)
        color_man = self.get_indicator_color("manhattan", manhattan)
        result_html = f"""
        <div style='text-align:center;'>
            <span style='background-color:{color_euc}; padding:8px; margin:4px;'>Евклидово расстояние: {euclidean:.4f}</span><br>
            <span style='background-color:{color_cos}; padding:8px; margin:4px;'>Косинусная схожесть: {cosine_similarity:.4f}</span><br>
            <span style='background-color:{color_man}; padding:8px; margin:4px;'>Манхэттенское расстояние: {manhattan:.4f}</span>
        </div>
        """
        self.results_label.setText(result_html)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = СравнениеИзображений()
    window.show()
    sys.exit(app.exec_())
