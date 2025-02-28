# interface_prototype.py
import sys
import os
import glob
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QFileDialog,
                             QHBoxLayout, QGroupBox, QCheckBox, QTextEdit, QScrollArea)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
import torch
import torchvision.transforms as transforms
from model_AE import ResidualAutoEncoder, get_embedding
from skimage.metrics import structural_similarity as ssim
import cv2

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))

def pearson_correlation(vec1, vec2):
    if np.std(vec1)==0 or np.std(vec2)==0:
        return 0.0
    return np.corrcoef(vec1, vec2)[0, 1]

def mse_metric(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

def ssim_metric(img1, img2):
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1
    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img2_gray = img2
    score, _ = ssim(img1_gray, img2_gray, full=True, data_range=255)
    return score

def histogram_intersection(img1, img2):
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1
    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img2_gray = img2
    hist1 = cv2.calcHist([img1_gray], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2_gray], [0], None, [256], [0, 256])
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    intersection = np.sum(np.minimum(hist1, hist2))
    return float(intersection)

class ImageComparator(QMainWindow):
    """
    Основной класс приложения для сравнения изображений.
    Позволяет загружать два изображения, выбирать метрики для их сравнения,
    вычислять метрики, а также находить топ-3 похожих изображения из папки images.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AEComparator')
        self.setGeometry(100, 100, 1200, 800)
        self.image1 = None
        self.image2 = None
        self.image1_np = None  # Для метрик без нейросети
        self.image2_np = None
        self.model = None
        self.init_model()
        self.init_ui()
        
    def init_model(self):
        """
        Загружает обученную модель автоэнкодера.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResidualAutoEncoder().to(device)
        try:
            self.model.load_state_dict(torch.load("ae_model.pth", map_location=device))
            self.model.eval()
        except Exception as e:
            print("Не удалось загрузить модель ae_model.pth. Убедитесь, что модель обучена.")
            self.model = None

    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # Раздел для отображения изображений
        image_layout = QHBoxLayout()
        self.image1_display = QLabel("Изображение 1")
        self.image1_display.setFixedSize(256, 256)
        self.image1_display.setAlignment(Qt.AlignCenter)
        
        self.image2_display = QLabel("Изображение 2")
        self.image2_display.setFixedSize(256, 256)
        self.image2_display.setAlignment(Qt.AlignCenter)
        
        image_layout.addWidget(self.image1_display)
        image_layout.addWidget(self.image2_display)
        
        load_button_layout = QHBoxLayout()
        load_button1 = QPushButton("Загрузить изображение 1")
        load_button1.clicked.connect(self.load_image1)
        load_button2 = QPushButton("Загрузить изображение 2")
        load_button2.clicked.connect(self.load_image2)
        load_button_layout.addWidget(load_button1)
        load_button_layout.addWidget(load_button2)
        
        self.embedding_metrics_group = QGroupBox("Метрики на основе эмбеддингов (НС)")
        emb_layout = QVBoxLayout()
        self.cb_cosine = QCheckBox("Косинусное сходство")
        self.cb_manhattan = QCheckBox("Манхэттенское расстояние")
        self.cb_pearson = QCheckBox("Корреляция Пирсона")
        # По умолчанию выбираем все
        self.cb_cosine.setChecked(True)
        self.cb_manhattan.setChecked(True)
        self.cb_pearson.setChecked(True)
        emb_layout.addWidget(self.cb_cosine)
        emb_layout.addWidget(self.cb_manhattan)
        emb_layout.addWidget(self.cb_pearson)
        self.embedding_metrics_group.setLayout(emb_layout)
        
        self.image_metrics_group = QGroupBox("Метрики на основе изображений (без НС)")
        img_layout = QVBoxLayout()
        self.cb_mse = QCheckBox("Среднеквадратичная ошибка (MSE)")
        self.cb_ssim = QCheckBox("SSIM")
        self.cb_hist = QCheckBox("Гистограммное пересечение")
        self.cb_mse.setChecked(True)
        self.cb_ssim.setChecked(True)
        self.cb_hist.setChecked(True)
        img_layout.addWidget(self.cb_mse)
        img_layout.addWidget(self.cb_ssim)
        img_layout.addWidget(self.cb_hist)
        self.image_metrics_group.setLayout(img_layout)
        
        compute_button = QPushButton("Вычислить метрики")
        compute_button.clicked.connect(self.compute_metrics)
        
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        
        similar_button = QPushButton("Найти похожие изображения (топ-3)")
        similar_button.clicked.connect(self.find_similar_images)
        
        self.similar_vector_group = QGroupBox("Топ-3 по косинусному сходству")
        self.similar_vector_layout = QHBoxLayout()
        self.similar_vector_group.setLayout(self.similar_vector_layout)
        
        self.similar_ssim_group = QGroupBox("Топ-3 по SSIM")
        self.similar_ssim_layout = QHBoxLayout()
        self.similar_ssim_group.setLayout(self.similar_ssim_layout)
        
        main_layout.addLayout(image_layout)
        main_layout.addLayout(load_button_layout)
        main_layout.addWidget(self.embedding_metrics_group)
        main_layout.addWidget(self.image_metrics_group)
        main_layout.addWidget(compute_button)
        main_layout.addWidget(self.results_display)
        main_layout.addWidget(similar_button)
        main_layout.addWidget(self.similar_vector_group)
        main_layout.addWidget(self.similar_ssim_group)
        
        container = QWidget()
        container.setLayout(main_layout)
        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)
        
    def load_image1(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec_():
            image_path = file_dialog.selectedFiles()[0]
            self.display_image(image_path, self.image1_display)
            self.image1 = image_path
            self.image1_np = np.array(Image.open(image_path).convert("RGB"))
            
    def load_image2(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec_():
            image_path = file_dialog.selectedFiles()[0]
            self.display_image(image_path, self.image2_display)
            self.image2 = image_path
            self.image2_np = np.array(Image.open(image_path).convert("RGB"))
            
    def display_image(self, image_path, label):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        
    def compute_metrics(self):
        if self.image1 is None or self.image2 is None:
            self.results_display.setText("Пожалуйста, загрузите оба изображения.")
            return
        
        results = ""
        
        
        transform_emb = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        img1_tensor = transform_emb(Image.open(self.image1).convert("RGB"))
        img2_tensor = transform_emb(Image.open(self.image2).convert("RGB"))
        
        if self.model is not None:
            embedding1 = get_embedding(self.model, img1_tensor)
            embedding2 = get_embedding(self.model, img2_tensor)
            emb1 = embedding1.cpu().numpy()
            emb2 = embedding2.cpu().numpy()
            
            results += "Метрики на основе эмбеддингов (НС):\n"
            if self.cb_cosine.isChecked():
                cos_sim = cosine_similarity(emb1, emb2)
                results += f"Косинусное сходство: {cos_sim:.4f}\n"
            if self.cb_manhattan.isChecked():
                manh = manhattan_distance(emb1, emb2)
                results += f"Манхэттенское расстояние: {manh:.4f}\n"
            if self.cb_pearson.isChecked():
                corr = pearson_correlation(emb1, emb2)
                results += f"Корреляция Пирсона: {corr:.4f}\n"
            results += "\n"
        else:
            results += "Модель автоэнкодера не загружена. Метрики на основе эмбеддингов недоступны.\n\n"
        
        img1 = np.array(Image.open(self.image1).resize((256,256))).astype(np.float32)
        img2 = np.array(Image.open(self.image2).resize((256,256))).astype(np.float32)
        
        results += "Метрики на основе изображений (без НС):\n"
        if self.cb_mse.isChecked():
            mse_val = mse_metric(img1, img2)
            results += f"MSE: {mse_val:.4f}\n"
        if self.cb_ssim.isChecked():
            ssim_val = ssim_metric(img1, img2)
            results += f"SSIM: {ssim_val:.4f}\n"
        if self.cb_hist.isChecked():
            hist_int = histogram_intersection(img1, img2)
            results += f"Гистограммное пересечение: {hist_int:.4f}\n"
            
        self.results_display.setText(results)
        
    def find_similar_images(self):
        """
        Находит топ-3 похожих изображения из выбранной пользователем папки
        по двум метрикам: косинусное сходство (между эмбеддингами) и SSIM.
        """
        if self.image1 is None:
            self.results_display.setText("Пожалуйста, загрузите изображение 1 для поиска похожих изображений.")
            return

        folder_path = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if not folder_path:
            self.results_display.setText("Папка не выбрана.")
            return

        transform_emb = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        query_tensor = transform_emb(Image.open(self.image1).convert("RGB"))
        query_embedding = None
        if self.model is not None:
            query_embedding = get_embedding(self.model, query_tensor).cpu().numpy()

        query_img_ssim = np.array(Image.open(self.image1).convert("RGB").resize((256, 256))).astype(np.float32)

        vector_results = []  
        ssim_results = []    

        valid_ext = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
        candidate_files = []
        for ext in valid_ext:
            candidate_files.extend(glob.glob(os.path.join(folder_path, ext)))

        for cand_path in candidate_files:
            if os.path.abspath(cand_path) == os.path.abspath(self.image1):
                continue

            if query_embedding is not None:
                try:
                    cand_tensor = transform_emb(Image.open(cand_path).convert("RGB"))
                    cand_embedding = get_embedding(self.model, cand_tensor).cpu().numpy()
                    cos_sim = cosine_similarity(query_embedding, cand_embedding)
                except Exception:
                    cos_sim = -np.inf
                vector_results.append((cand_path, cos_sim))

            # SSIM
            try:
                cand_img_ssim = np.array(Image.open(cand_path).convert("RGB").resize((256, 256))).astype(np.float32)
                ssim_score = ssim_metric(query_img_ssim, cand_img_ssim)
            except Exception:
                ssim_score = -1
            ssim_results.append((cand_path, ssim_score))

        vector_results.sort(key=lambda x: x[1], reverse=True)
        ssim_results.sort(key=lambda x: x[1], reverse=True)

        top_vector = vector_results[:3]
        top_ssim = ssim_results[:3]

        self.clear_layout(self.similar_vector_layout)
        self.clear_layout(self.similar_ssim_layout)

        for path, sim in top_vector:
            widget = self.create_result_widget(path, f"Косинусное сходство: {sim:.4f}")
            self.similar_vector_layout.addWidget(widget)

        for path, score in top_ssim:
            widget = self.create_result_widget(path, f"SSIM: {score:.4f}")
            self.similar_ssim_layout.addWidget(widget)

            
    def create_result_widget(self, image_path, text):
        """
        Создает виджет для отображения миниатюры изображения и подписи с метрикой.
        """
        container = QWidget()
        layout = QVBoxLayout()
        img_label = QLabel()
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignCenter)
        text_label = QLabel(text)
        text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(img_label)
        layout.addWidget(text_label)
        container.setLayout(layout)
        return container
    
    def clear_layout(self, layout):
        """
        Удаляет все виджеты из заданного layout.
        """
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageComparator()
    window.show()
    sys.exit(app.exec_())
