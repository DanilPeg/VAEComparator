import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QFileDialog, QComboBox, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
import numpy as np

class ImageComparator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('VAEComparator-test')
        self.setGeometry(100, 100, 800, 600)

        self.image1 = None
        self.image2 = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        image_layout = QHBoxLayout()
        self.image1_label = QLabel("Изображение 1:")
        self.image1_display = QLabel(self)
        self.image1_display.setFixedSize(256, 256)
        self.image1_display.setAlignment(Qt.AlignCenter)

        self.image2_label = QLabel("Изображение 2:")
        self.image2_display = QLabel(self)
        self.image2_display.setFixedSize(256, 256)
        self.image2_display.setAlignment(Qt.AlignCenter)

        image_layout.addWidget(self.image1_label)
        image_layout.addWidget(self.image1_display)
        image_layout.addWidget(self.image2_label)
        image_layout.addWidget(self.image2_display)

        load_button1 = QPushButton("Загрузить первое изображение")
        load_button1.clicked.connect(self.load_image1)

        load_button2 = QPushButton("Загрузить второе изображение")
        load_button2.clicked.connect(self.load_image2)

        self.metric_combo = QComboBox()
        self.metric_combo.addItem("Метрика 1")
        self.metric_combo.addItem("Метрика 2")
        self.metric_combo.addItem("Метрика 3")

        compute_button = QPushButton("Вычислить метрику")
        compute_button.clicked.connect(self.compute_metric)

        self.metric_value_label = QLabel("Метрика близости: ")

        layout.addLayout(image_layout)
        layout.addWidget(load_button1)
        layout.addWidget(load_button2)
        layout.addWidget(self.metric_combo)
        layout.addWidget(compute_button)
        layout.addWidget(self.metric_value_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image1(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec_():
            image_path = file_dialog.selectedFiles()[0]
            self.display_image(image_path, self.image1_display)
            self.image1 = np.array(Image.open(image_path))

    def load_image2(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        if file_dialog.exec_():
            image_path = file_dialog.selectedFiles()[0]
            self.display_image(image_path, self.image2_display)
            self.image2 = np.array(Image.open(image_path))

    def display_image(self, image_path, label):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)

    def compute_metric(self):
        if self.image1 is not None and self.image2 is not None:
            metric_value = np.random.rand() 
            self.metric_value_label.setText(f"Метрика близости: {metric_value:.4f}")
        else:
            self.metric_value_label.setText("Загрузите оба изображения")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageComparator()
    window.show()
    sys.exit(app.exec_())
