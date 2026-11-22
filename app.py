import sys
import os
import cv2
import json
import csv
import tempfile
import random
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFileDialog, QSlider, QGridLayout, QGroupBox,
    QTabWidget, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar, 
    QStatusBar, QCheckBox, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QFrame, QTextEdit, QListWidget, QListWidgetItem,
    QDialog
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QSize, QMimeData
from PySide6.QtGui import (QImage, QPixmap, QFont, QColor, QKeySequence, QIcon,
                           QDragEnterEvent, QDropEvent, QShortcut)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

# ==================== Global Dark Theme ====================
DARK_STYLESHEET = """
    QMainWindow, QWidget { background-color: #1e1e1e; color: #e0e0e0; }
    QGroupBox { border: 1px solid #3a3a3a; border-radius: 4px; margin-top: 10px; padding-top: 10px; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #4a9eff; }
    QPushButton { background-color: #2d2d2d; border: 1px solid #3a3a3a; border-radius: 4px; padding: 6px 12px; }
    QPushButton:hover { background-color: #3a3a3a; border: 1px solid #4a9eff; }
    QPushButton:pressed { background-color: #4a9eff; color: #1e1e1e; }
    QPushButton:disabled { color: #555; background-color: #252525; }
    QComboBox, QSpinBox, QDoubleSpinBox { background-color: #2d2d2d; border: 1px solid #3a3a3a; border-radius: 4px; padding: 4px; }
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border: 1px solid #4a9eff; }
    QComboBox::drop-down { border: 0px; width: 20px; }
    QSlider::groove:horizontal { border: 1px solid #3a3a3a; height: 4px; background: #2d2d2d; }
    QSlider::handle:horizontal { background: #4a9eff; width: 12px; margin: -5px 0; border-radius: 6px; }
    QSlider::sub-page:horizontal { background: #4a9eff; }
    QTableWidget { background-color: #1e1e1e; gridline-color: #3a3a3a; border: 1px solid #3a3a3a; }
    QHeaderView::section { background-color: #2d2d2d; padding: 4px; border: 1px solid #3a3a3a; color: #4a9eff; }
    QTabWidget::pane { border: 1px solid #3a3a3a; }
    QTabBar::tab { background: #2d2d2d; padding: 8px 16px; border: 1px solid #3a3a3a; border-bottom: none; }
    QTabBar::tab:selected { background: #1e1e1e; border-bottom: 2px solid #4a9eff; }
    QLabel { background-color: transparent; }
    QTextEdit { background-color: #1e1e1e; border: 1px solid #3a3a3a; color: #e0e0e0; }
    QCheckBox { spacing: 8px; }
    QListWidget { background-color: #1e1e1e; border: 1px solid #3a3a3a; }
    QListWidget::item:selected { background-color: #4a9eff; color: #1e1e1e; }
"""

# ==================== STN Model ====================
class STNModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(STNModel, self).__init__()
        self.in_channels = in_channels
        
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((3, 3)) 
        )
        
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x, theta, grid

    def forward(self, x):
        x_transformed, theta, grid = self.stn(x)
        x = F.relu(F.max_pool2d(self.conv1(x_transformed), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = nn.AdaptiveAvgPool2d((5, 5))(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x_transformed, theta, grid

# ==================== Training Thread ====================
class TrainingThread(QThread):
    loss_signal = Signal(float)
    acc_signal = Signal(float)
    epoch_signal = Signal(int)
    lr_signal = Signal(float)
    image_signal = Signal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(self, model, device, train_loader, test_loader, epochs, learning_rate, optimizer_type='sgd', scheduler_type='none', use_aug=False, early_stopping_patience=0):
        super().__init__()
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.running = True
        self.patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0

    def run(self):
        try:
            if self.optimizer_type == 'sgd':
                optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
            elif self.optimizer_type == 'adam':
                optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            else:
                optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

            if self.scheduler_type == 'step':
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
            elif self.scheduler_type == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
            else:
                scheduler = None

            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

            for epoch in range(1, self.epochs + 1):
                if not self.running: break
                self.epoch_signal.emit(epoch)
                self.lr_signal.emit(optimizer.param_groups[0]['lr'])
                
                self.model.train()
                train_loss, correct, total = 0, 0, 0
                
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    if not self.running: break
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output, transformed, theta, grid = self.model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += len(data)
                    
                    if batch_idx % 50 == 0:
                        self.loss_signal.emit(train_loss / (batch_idx + 1))
                        self.acc_signal.emit(100. * correct / total)
                        self.emit_images(data[0], transformed[0], theta[0], grid[0])

                if scheduler: scheduler.step()

                val_loss, val_acc = 0, 0
                if self.test_loader is not None and self.running:
                    self.model.eval()
                    correct, total = 0, 0
                    with torch.no_grad():
                        for data, target in self.test_loader:
                            data, target = data.to(self.device), target.to(self.device)
                            output, _, _, _ = self.model(data)
                            val_loss += F.nll_loss(output, target, reduction='sum').item()
                            pred = output.argmax(dim=1, keepdim=True)
                            correct += pred.eq(target.view_as(pred)).sum().item()
                            total += len(data)
                    val_loss /= total
                    val_acc = 100. * correct / total
                    
                    self.loss_signal.emit(val_loss)
                    self.acc_signal.emit(val_acc)
                    
                    if self.patience > 0:
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.best_model_state = self.model.state_dict()
                            self.patience_counter = 0
                        else:
                            self.patience_counter += 1
                            if self.patience_counter >= self.patience:
                                self.running = False
                
            if self.best_model_state:
                self.model.load_state_dict(self.best_model_state)
                
            self.finished_signal.emit(history)
        except Exception as e:
            self.error_signal.emit(str(e))

    def emit_images(self, input_t, trans_t, theta_t, grid_t):
        input_img = input_t.cpu().numpy().squeeze()
        if input_img.ndim == 3: input_img = np.transpose(input_img, (1, 2, 0))
        transformed_img = trans_t.cpu().numpy().squeeze()
        if transformed_img.ndim == 3: transformed_img = np.transpose(transformed_img, (1, 2, 0))
        self.image_signal.emit(input_img, transformed_img, theta_t.cpu().numpy(), grid_t.cpu().numpy())

    def stop(self):
        self.running = False
        self.wait()

# ==================== Plot Widget ====================
class TrainingPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 4), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(121)
        self.ax2 = self.figure.add_subplot(122)
        self.train_losses, self.train_accs, self.val_losses, self.val_accs = [], [], [], []
        self.setup_plots()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def setup_plots(self):
        for ax, title in [(self.ax1, 'Loss'), (self.ax2, 'Accuracy (%)')]:
            ax.set_title(title, color='white')
            ax.set_xlabel('Epoch', color='white')
            ax.tick_params(colors='white')
            ax.grid(True, color='#3a3a3a')
            ax.set_facecolor('#2d2d2d')
        self.figure.tight_layout()
        
    def update_metrics(self, loss, accuracy, is_validation=False):
        if is_validation: self.val_losses.append(loss); self.val_accs.append(accuracy)
        else: self.train_losses.append(loss); self.train_accs.append(accuracy)
        self.update_plots()
        
    def update_plots(self):
        self.ax1.clear(); self.ax2.clear()
        if self.train_losses:
            epochs = range(1, len(self.train_losses) + 1)
            self.ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
            self.ax2.plot(epochs, self.train_accs, 'b-', label='Train Acc')
        if self.val_losses:
            epochs = range(1, len(self.val_losses) + 1)
            self.ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
            self.ax2.plot(epochs, self.val_accs, 'r-', label='Val Acc')
        self.setup_plots()
        self.ax1.legend(facecolor='#2d2d2d', edgecolor='#3a3a3a')
        self.ax2.legend(facecolor='#2d2d2d', edgecolor='#3a3a3a')
        self.canvas.draw()
        
    def clear_data(self):
        self.train_losses, self.train_accs, self.val_losses, self.val_accs = [], [], [], []
        self.update_plots()

# ==================== Manual Transform Tab ====================
class ManualTransformTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.dataset_paths = []
        self.current_index = -1
        self.current_original_img = None
        self.transform_params = {
            'translate_x': 0, 'translate_y': 0, 'rotation': 0,
            'scale_x': 1.0, 'scale_y': 1.0, 'shear_x': 0, 'shear_y': 0
        }
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        images_layout = QHBoxLayout()
        
        self.original_label = self.create_image_label("Original Image\n(Drag & Drop Image Here)")
        self.transformed_label = self.create_image_label("Transformed Image")
        images_layout.addWidget(self.original_label)
        images_layout.addWidget(self.transformed_label)
        left_layout.addLayout(images_layout)
        
        nav_layout = QHBoxLayout()
        self.btn_load = QPushButton("Load Dataset Folder")
        self.btn_example = QPushButton("Load Example Images") # NEW
        self.btn_prev = QPushButton("<< Previous (←)")
        self.btn_next = QPushButton("Next (→) >>")
        self.btn_save = QPushButton("Save Transformed")
        self.btn_batch = QPushButton("Batch Process")
        self.btn_save_profile = QPushButton("Save Profile")
        self.btn_load_profile = QPushButton("Load Profile")
        
        self.btn_load.clicked.connect(self.load_dataset)
        self.btn_example.clicked.connect(self.load_example_dataset) # NEW
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_batch.clicked.connect(self.batch_process)
        self.btn_save_profile.clicked.connect(self.save_profile)
        self.btn_load_profile.clicked.connect(self.load_profile)
        
        nav_layout.addWidget(self.btn_load)
        nav_layout.addWidget(self.btn_example) # NEW
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(self.btn_save)
        nav_layout.addWidget(self.btn_batch)
        nav_layout.addWidget(self.btn_save_profile)
        nav_layout.addWidget(self.btn_load_profile)
        left_layout.addLayout(nav_layout)
        
        self.lbl_status = QLabel("No dataset loaded. Click 'Load Example Images' to try instantly.")
        left_layout.addWidget(self.lbl_status)
        
        self.lbl_matrix = QTextEdit("Affine Matrix (2x3):\n[[1.00, 0.00, 0.00]\n [0.00, 1.00, 0.00]]")
        self.lbl_matrix.setReadOnly(True)
        self.lbl_matrix.setMaximumHeight(80)
        left_layout.addWidget(self.lbl_matrix)
        
        top_layout.addWidget(left_widget)
        
        controls_group = QGroupBox("Spatial Transformation Controls")
        controls_layout = QGridLayout()
        
        controls_layout.addWidget(QLabel("Transform Type:"), 0, 0)
        self.combo_transform = QComboBox()
        self.combo_transform.addItems(["Affine", "Perspective", "Elastic", "Custom"])
        controls_layout.addWidget(self.combo_transform, 0, 1)
        
        self.sliders = {}
        params = [
            ("Translate X", -100, 100, 0, "translate_x"),
            ("Translate Y", -100, 100, 0, "translate_y"),
            ("Rotation (Deg)", -180, 180, 0, "rotation"),
            ("Scale X", 10, 200, 100, "scale_x"),
            ("Scale Y", 10, 200, 100, "scale_y"),
            ("Shear X", -45, 45, 0, "shear_x"),
            ("Shear Y", -45, 45, 0, "shear_y")
        ]
        
        for row, (name, min_val, max_val, default, key) in enumerate(params):
            lbl_name = QLabel(name)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default)
            lbl_value = QLabel(str(default))
            lbl_value.setMinimumWidth(40)
            slider.valueChanged.connect(lambda v, n=name, lbl=lbl_value, k=key: self.on_slider_change(n, v, lbl, k))
            controls_layout.addWidget(lbl_name, row + 1, 0)
            controls_layout.addWidget(slider, row + 1, 1)
            controls_layout.addWidget(lbl_value, row + 1, 2)
            self.sliders[key] = slider
        
        controls_layout.addWidget(QLabel("Interpolation:"), len(params) + 1, 0)
        self.combo_interp = QComboBox()
        self.combo_interp.addItems(["Nearest", "Bilinear", "Bicubic"])
        controls_layout.addWidget(self.combo_interp, len(params) + 1, 1)
        
        self.btn_reset = QPushButton("Reset Transformations")
        self.btn_reset.clicked.connect(self.reset_transformations)
        controls_layout.addWidget(self.btn_reset, len(params) + 2, 0, 1, 3)
        
        controls_group.setLayout(controls_layout)
        top_layout.addWidget(controls_group)
        
        splitter.addWidget(top_widget)
        
        thumb_group = QGroupBox("Dataset Navigator")
        thumb_layout = QVBoxLayout()
        self.thumb_list = QListWidget()
        self.thumb_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.thumb_list.setIconSize(QSize(100, 100))
        self.thumb_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.thumb_list.setMovement(QListWidget.Movement.Static)
        self.thumb_list.currentRowChanged.connect(self.on_thumb_changed)
        thumb_layout.addWidget(self.thumb_list)
        thumb_group.setLayout(thumb_layout)
        
        splitter.addWidget(thumb_group)
        splitter.setSizes([600, 200])
        
        layout.addWidget(splitter)
        self.setLayout(layout)
        
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self.next_image)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self.prev_image)
        
    def create_image_label(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("background-color: #2b2b2b; color: #888; border: 1px solid #555;")
        lbl.setMinimumSize(400, 400)
        return lbl
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.dataset_paths = [path]
                self.current_index = 0
                self.display_current()
                self.lbl_status.setText(f"Loaded via drag & drop: {os.path.basename(path)}")
                
    def load_example_dataset(self):
        """Generates a synthetic dataset of shapes/numbers for testing."""
        example_dir = os.path.join(tempfile.gettempdir(), "stn_example_data")
        os.makedirs(example_dir, exist_ok=True)
        
        # Clear old examples
        for f in os.listdir(example_dir):
            os.remove(os.path.join(example_dir, f))
            
        for i in range(15):
            img = np.ones((64, 64, 3), dtype=np.uint8) * 50  # Dark background
            text = str(random.randint(0, 9))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text, (15, 50), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Apply random rotation to make it interesting for STN
            angle = random.uniform(-45, 45)
            M = cv2.getRotationMatrix2D((32, 32), angle, 1)
            img = cv2.warpAffine(img, M, (64, 64), borderMode=cv2.BORDER_CONSTANT, borderValue=(50, 50, 50))
            
            cv2.imwrite(os.path.join(example_dir, f"example_{i}.png"), img)
            
        self.load_dataset(example_dir)
        
    def load_dataset(self, folder=None):
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
            if not folder: return
            
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        self.dataset_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_ext)]
        if not self.dataset_paths:
            self.lbl_status.setText("No valid images found.")
            return
            
        self.current_index = 0
        self.lbl_status.setText(f"Loaded {len(self.dataset_paths)} images.")
        
        self.thumb_list.clear()
        for path in self.dataset_paths:
            item = QListWidgetItem(os.path.basename(path))
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[2] * rgb.shape[1], QImage.Format.Format_RGB888)
                item.setIcon(QIcon(QPixmap.fromImage(qimg)))
            self.thumb_list.addItem(item)
            
        self.display_current()
        
    def on_thumb_changed(self, row):
        if row >= 0 and row < len(self.dataset_paths):
            self.current_index = row
            self.display_current()
        
    def display_current(self):
        if not self.dataset_paths: return
        path = self.dataset_paths[self.current_index]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None: return
        self.current_original_img = img
        self.update_images()
        self.lbl_status.setText(f"Viewing {self.current_index + 1}/{len(self.dataset_paths)}: {os.path.basename(path)}")
        
    def next_image(self):
        if self.dataset_paths and self.current_index < len(self.dataset_paths) - 1:
            self.current_index += 1
            self.thumb_list.setCurrentRow(self.current_index)
            
    def prev_image(self):
        if self.dataset_paths and self.current_index > 0:
            self.current_index -= 1
            self.thumb_list.setCurrentRow(self.current_index)
            
    def on_slider_change(self, name, value, label, key):
        if "Scale" in name:
            label.setText(f"{value/100.0:.2f}")
            self.transform_params[key] = value / 100.0
        else:
            label.setText(str(value))
            self.transform_params[key] = value
        self.update_images()
        
    def reset_transformations(self):
        defaults = {'translate_x': 0, 'translate_y': 0, 'rotation': 0, 'scale_x': 100, 'scale_y': 100, 'shear_x': 0, 'shear_y': 0}
        for key, val in defaults.items():
            self.sliders[key].setValue(val)
            
    def save_profile(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Transform Profile", "", "JSON Files (*.json)")
        if path:
            with open(path, 'w') as f:
                json.dump(self.transform_params, f)
            self.lbl_status.setText(f"Saved profile to {path}")
            
    def load_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Transform Profile", "", "JSON Files (*.json)")
        if path:
            with open(path, 'r') as f:
                params = json.load(f)
            for key, val in params.items():
                if key in self.sliders:
                    if "scale" in key:
                        self.sliders[key].setValue(int(val * 100))
                    else:
                        self.sliders[key].setValue(val)
                        
    def apply_spatial_transform(self, img):
        rows, cols = img.shape[:2]
        tx = self.transform_params['translate_x']
        ty = self.transform_params['translate_y']
        angle = self.transform_params['rotation']
        sx = self.transform_params['scale_x']
        sy = self.transform_params['scale_y']
        shx = self.transform_params['shear_x'] / 100.0
        shy = self.transform_params['shear_y'] / 100.0
        
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        cx, cy = cols / 2.0, rows / 2.0
        
        def transform_point(x, y):
            x -= cx; y -= cy
            x *= sx; y *= sy
            x_new = x + shx * y; y_new = y + shy * x
            x, y = x_new, y_new
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            return [x_rot + cx + tx, y_rot + cy + ty]
            
        pts_src = np.float32([[cx, cy], [cx, 0], [cols, cy]])
        pts_dst = np.float32([transform_point(cx, cy), transform_point(cx, 0), transform_point(cols, cy)])
        M = cv2.getAffineTransform(pts_src, pts_dst)
        
        m_str = np.array2string(M, precision=2, separator=', ')
        self.lbl_matrix.setText(f"Affine Matrix (2x3):\n{m_str}")
        
        transformed_img = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return transformed_img
        
    def update_images(self):
        if self.current_original_img is None: return
        original_qimg = self.cv2_to_qimage(self.current_original_img)
        self.original_label.setPixmap(QPixmap.fromImage(original_qimg).scaled(
            self.original_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        transformed_cv = self.apply_spatial_transform(self.current_original_img)
        transformed_qimg = self.cv2_to_qimage(transformed_cv)
        self.transformed_label.setPixmap(QPixmap.fromImage(transformed_qimg).scaled(
            self.transformed_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            
    def save_image(self):
        if self.current_original_img is None: return
        transformed_cv = self.apply_spatial_transform(self.current_original_img)
        original_path = self.dataset_paths[self.current_index]
        save_dir = os.path.join(os.path.dirname(original_path), "transformed_output")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"st_{os.path.basename(original_path)}")
        cv2.imwrite(save_path, transformed_cv)
        self.lbl_status.setText(f"Saved: {save_path}")
        
    def batch_process(self):
        if not self.dataset_paths:
            QMessageBox.warning(self, "Warning", "No dataset loaded!")
            return
        save_dir = os.path.join(os.path.dirname(self.dataset_paths[0]), "batch_output")
        os.makedirs(save_dir, exist_ok=True)
        for i, path in enumerate(self.dataset_paths):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                transformed = self.apply_spatial_transform(img)
                cv2.imwrite(os.path.join(save_dir, f"st_{os.path.basename(path)}"), transformed)
        QMessageBox.information(self, "Success", f"Processed {len(self.dataset_paths)} images!")
        
    def cv2_to_qimage(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        return QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)

# ==================== Confusion Matrix Dialog ====================
class ConfusionMatrixDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confusion Matrix")
        self.resize(600, 600)
        layout = QVBoxLayout()
        self.figure = Figure(facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_matrix(self, matrix, classes):
        self.ax.clear()
        im = self.ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        self.ax.figure.colorbar(im, ax=self.ax)
        self.ax.set(xticks=np.arange(matrix.shape[1]),
                   yticks=np.arange(matrix.shape[0]),
                   xticklabels=classes, yticklabels=classes,
                   title="Confusion Matrix",
                   ylabel="True Label",
                   xlabel="Predicted Label")
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        thresh = matrix.max() / 2.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                self.ax.text(j, i, format(matrix[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if matrix[i, j] > thresh else "black")
        self.figure.tight_layout()
        self.canvas.draw()

# ==================== Training Tab ====================
class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = STNModel().to(self.device)
        self.train_loader = None
        self.test_loader = None
        self.training_thread = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Dataset Group
        ds_group = QGroupBox("Dataset Selection")
        ds_layout = QGridLayout()
        ds_layout.addWidget(QLabel("Dataset:"), 0, 0)
        self.combo_dataset = QComboBox()
        self.combo_dataset.addItems(["MNIST (1-Channel)", "CIFAR10 (3-Channel)", "Custom ImageFolder"])
        self.combo_dataset.currentIndexChanged.connect(self.toggle_custom_dataset_ui)
        ds_layout.addWidget(self.combo_dataset, 0, 1)
        
        self.btn_load_custom = QPushButton("Select Custom Folder")
        self.btn_load_custom.setEnabled(False)
        self.btn_load_custom.clicked.connect(self.load_custom_dataset)
        ds_layout.addWidget(self.btn_load_custom, 1, 0, 1, 2)
        
        ds_btn_layout = QHBoxLayout()
        self.btn_load_data = QPushButton("Load/Download Selected Dataset")
        self.btn_load_data.clicked.connect(self.load_builtin_dataset)
        
        # NEW: Quick Example Button
        self.btn_example_train = QPushButton("Run 1-Epoch MNIST Example")
        self.btn_example_train.setStyleSheet("QPushButton { background-color: #2d5a2d; border: 1px solid #4a9eff; } QPushButton:hover { background-color: #3a7a3a; }")
        self.btn_example_train.clicked.connect(self.run_example_training)
        
        ds_btn_layout.addWidget(self.btn_load_data)
        ds_btn_layout.addWidget(self.btn_example_train)
        ds_layout.addLayout(ds_btn_layout, 2, 0, 1, 2)
        
        aug_layout = QHBoxLayout()
        self.chk_rot = QCheckBox("Random Rotation")
        self.chk_affine = QCheckBox("Random Affine")
        self.chk_erase = QCheckBox("Random Erasing")
        aug_layout.addWidget(self.chk_rot)
        aug_layout.addWidget(self.chk_affine)
        aug_layout.addWidget(self.chk_erase)
        ds_layout.addLayout(aug_layout, 3, 0, 1, 2)
        ds_group.setLayout(ds_layout)
        left_layout.addWidget(ds_group)
        
        # Params Group
        params_group = QGroupBox("Training Parameters")
        params_layout = QGridLayout()
        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.spin_epochs = QSpinBox(); self.spin_epochs.setRange(1, 100); self.spin_epochs.setValue(10)
        params_layout.addWidget(self.spin_epochs, 0, 1)
        
        params_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.spin_batch_size = QSpinBox(); self.spin_batch_size.setRange(1, 1024); self.spin_batch_size.setValue(64)
        params_layout.addWidget(self.spin_batch_size, 1, 1)
        
        params_layout.addWidget(QLabel("Num Workers:"), 2, 0)
        self.spin_workers = QSpinBox(); self.spin_workers.setRange(0, 16); self.spin_workers.setValue(4)
        params_layout.addWidget(self.spin_workers, 2, 1)
        
        params_layout.addWidget(QLabel("Learning Rate:"), 3, 0)
        self.spin_lr = QDoubleSpinBox(); self.spin_lr.setRange(0.0001, 1.0); self.spin_lr.setValue(0.01); self.spin_lr.setSingleStep(0.001)
        params_layout.addWidget(self.spin_lr, 3, 1)
        
        params_layout.addWidget(QLabel("Optimizer:"), 4, 0)
        self.combo_optimizer = QComboBox(); self.combo_optimizer.addItems(["SGD", "Adam", "RMSprop"])
        params_layout.addWidget(self.combo_optimizer, 4, 1)
        
        params_layout.addWidget(QLabel("Scheduler:"), 5, 0)
        self.combo_scheduler = QComboBox(); self.combo_scheduler.addItems(["None", "StepLR", "CosineAnnealing"])
        params_layout.addWidget(self.combo_scheduler, 5, 1)
        
        self.chk_early_stop = QCheckBox("Enable Early Stopping")
        params_layout.addWidget(self.chk_early_stop, 6, 0, 1, 2)
        params_layout.addWidget(QLabel("Patience:"), 7, 0)
        self.spin_patience = QSpinBox(); self.spin_patience.setRange(1, 10); self.spin_patience.setValue(3)
        params_layout.addWidget(self.spin_patience, 7, 1)
        
        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)
        
        # Control Buttons
        ctrl_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Training"); self.btn_start.clicked.connect(self.start_training)
        self.btn_stop = QPushButton("Stop Training"); self.btn_stop.clicked.connect(self.stop_training); self.btn_stop.setEnabled(False)
        self.btn_save_model = QPushButton("Save Model"); self.btn_save_model.clicked.connect(self.save_model)
        self.btn_export_logs = QPushButton("Export Logs CSV"); self.btn_export_logs.clicked.connect(self.export_logs)
        self.btn_model_summary = QPushButton("Model Summary"); self.btn_model_summary.clicked.connect(self.show_model_summary)
        self.btn_conf_matrix = QPushButton("Confusion Matrix"); self.btn_conf_matrix.clicked.connect(self.show_confusion_matrix)
        ctrl_layout.addWidget(self.btn_start); ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_save_model); ctrl_layout.addWidget(self.btn_export_logs)
        ctrl_layout.addWidget(self.btn_model_summary); ctrl_layout.addWidget(self.btn_conf_matrix)
        left_layout.addLayout(ctrl_layout)
        
        # Progress & Viz
        prog_group = QGroupBox("Training Progress & Visualization")
        prog_layout = QGridLayout()
        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 100)
        prog_layout.addWidget(self.progress_bar, 0, 0, 1, 4)
        
        self.lbl_epoch = QLabel("Epoch: 0/0 | LR: 0.0000")
        self.lbl_loss = QLabel("Loss: -")
        self.lbl_acc = QLabel("Acc: -")
        prog_layout.addWidget(self.lbl_epoch, 1, 0, 1, 4)
        prog_layout.addWidget(self.lbl_loss, 2, 0, 1, 2)
        prog_layout.addWidget(self.lbl_acc, 2, 2, 1, 2)
        
        self.lbl_input = self.create_img_label("Input")
        self.lbl_transformed = self.create_img_label("STN Output")
        self.lbl_theta = self.create_img_label("Theta Matrix")
        self.lbl_grid = self.create_img_label("Sampling Grid")
        
        prog_layout.addWidget(self.lbl_input, 3, 0)
        prog_layout.addWidget(self.lbl_transformed, 3, 1)
        prog_layout.addWidget(self.lbl_theta, 3, 2)
        prog_layout.addWidget(self.lbl_grid, 3, 3)
        prog_group.setLayout(prog_layout)
        left_layout.addWidget(prog_group)
        
        splitter.addWidget(left_widget)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.plot_widget = TrainingPlotWidget()
        right_layout.addWidget(self.plot_widget)
        
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(4)
        self.log_table.setHorizontalHeaderLabels(["Epoch", "Train Loss", "Train Acc", "Val Acc"])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        log_layout.addWidget(self.log_table)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        splitter.addWidget(right_widget)
        layout.addWidget(splitter)
        self.setLayout(layout)
        
    def create_img_label(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("background-color: #2b2b2b; color: #888; border: 1px solid #555;")
        lbl.setMinimumSize(150, 150)
        return lbl
        
    def toggle_custom_dataset_ui(self, index):
        self.btn_load_custom.setEnabled(index == 2)
        
    def get_transforms(self):
        t_list = [transforms.Resize((32, 32))]
        if self.chk_rot.isChecked(): t_list.append(transforms.RandomRotation(15))
        if self.chk_affine.isChecked(): t_list.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
        t_list.append(transforms.ToTensor())
        if self.chk_erase.isChecked(): t_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)))
        return transforms.Compose(t_list)
        
    def load_builtin_dataset(self):
        idx = self.combo_dataset.currentIndex()
        batch_size = self.spin_batch_size.value()
        num_workers = self.spin_workers.value()
        try:
            transform = self.get_transforms()
            if idx == 0: # MNIST
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
                test_set = datasets.MNIST(root='./data', train=False, transform=transform)
                self.model = STNModel(in_channels=1, num_classes=10).to(self.device)
            elif idx == 1: # CIFAR10
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
                test_set = datasets.CIFAR10(root='./data', train=False, transform=transform)
                self.model = STNModel(in_channels=3, num_classes=10).to(self.device)
                
            self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            QMessageBox.information(self, "Success", f"{self.combo_dataset.currentText()} loaded!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
    def run_example_training(self):
        """Quick start: Configures UI for a fast 1-epoch MNIST run and starts."""
        self.combo_dataset.setCurrentIndex(0) # MNIST
        self.spin_epochs.setValue(1)
        self.spin_batch_size.setValue(64)
        self.spin_lr.setValue(0.01)
        self.combo_optimizer.setCurrentText("SGD")
        
        # Check if MNIST is already loaded to avoid re-downloading unnecessarily
        if self.train_loader is None or self.model.in_channels != 1:
            self.load_builtin_dataset()
            
        if self.train_loader is not None:
            self.start_training()
            
    def load_custom_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select ImageFolder Directory")
        if folder:
            transform = self.get_transforms()
            try:
                full_dataset = datasets.ImageFolder(root=folder, transform=transform)
                train_size = int(0.8 * len(full_dataset))
                val_size = len(full_dataset) - train_size
                train_set, val_set = random_split(full_dataset, [train_size, val_size])
                
                self.train_loader = DataLoader(train_set, batch_size=self.spin_batch_size.value(), shuffle=True, num_workers=self.spin_workers.value())
                self.test_loader = DataLoader(val_set, batch_size=self.spin_batch_size.value(), shuffle=False, num_workers=self.spin_workers.value())
                
                self.model = STNModel(in_channels=3, num_classes=len(full_dataset.classes)).to(self.device)
                QMessageBox.information(self, "Success", f"Custom dataset loaded: {train_size} train, {val_size} val, {len(full_dataset.classes)} classes.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                
    def start_training(self):
        if self.train_loader is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return
        self.plot_widget.clear_data()
        self.log_table.setRowCount(0)
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.btn_example_train.setEnabled(False) # Disable example during training
        
        patience = self.spin_patience.value() if self.chk_early_stop.isChecked() else 0
        
        self.training_thread = TrainingThread(
            self.model, self.device, self.train_loader, self.test_loader,
            self.spin_epochs.value(), self.spin_lr.value(), 
            self.combo_optimizer.currentText().lower(),
            self.combo_scheduler.currentText().lower().replace('lr', '').replace('annealing', ''),
            self.chk_rot.isChecked() or self.chk_affine.isChecked(),
            patience
        )
        self.training_thread.loss_signal.connect(self.update_loss)
        self.training_thread.acc_signal.connect(self.update_acc)
        self.training_thread.epoch_signal.connect(self.update_epoch)
        self.training_thread.lr_signal.connect(self.update_lr)
        self.training_thread.image_signal.connect(self.update_images)
        self.training_thread.finished_signal.connect(self.train_finished)
        self.training_thread.error_signal.connect(self.train_error)
        self.training_thread.start()
        
    def stop_training(self):
        if self.training_thread:
            self.training_thread.stop()
            self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
            self.btn_example_train.setEnabled(True)
            
    def update_loss(self, loss):
        self.lbl_loss.setText(f"Loss: {loss:.4f}")
        if len(self.plot_widget.train_accs) > len(self.plot_widget.train_losses):
            self.plot_widget.train_losses.append(loss)
        self.plot_widget.update_plots()
        
    def update_acc(self, acc):
        self.lbl_acc.setText(f"Acc: {acc:.2f}%")
        if len(self.plot_widget.train_losses) > len(self.plot_widget.train_accs):
            self.plot_widget.train_accs.append(acc)
        else:
            self.plot_widget.val_losses.append(self.plot_widget.train_losses[-1])
            self.plot_widget.val_accs.append(acc)
            row = self.log_table.rowCount() - 1
            if row >= 0:
                self.log_table.setItem(row, 1, QTableWidgetItem(f"{self.plot_widget.train_losses[-1]:.4f}"))
                self.log_table.setItem(row, 2, QTableWidgetItem(f"{self.plot_widget.train_accs[-1]:.2f}%"))
                self.log_table.setItem(row, 3, QTableWidgetItem(f"{acc:.2f}%"))
        self.plot_widget.update_plots()
        
    def update_epoch(self, epoch):
        total = self.spin_epochs.value()
        self.lbl_epoch.setText(f"Epoch: {epoch}/{total} | LR: {self.spin_lr.value():.4f}")
        self.progress_bar.setValue(int((epoch / total) * 100))
        row = self.log_table.rowCount()
        self.log_table.insertRow(row)
        self.log_table.setItem(row, 0, QTableWidgetItem(str(epoch)))
        
    def update_lr(self, lr):
        current_text = self.lbl_epoch.text().split("|")[0]
        self.lbl_epoch.setText(f"{current_text} | LR: {lr:.4f}")
        
    def update_images(self, input_img, trans_img, theta_img, grid_img):
        self.set_pixmap(self.lbl_input, input_img)
        self.set_pixmap(self.lbl_transformed, trans_img)
        
        theta_str = "\n".join(["  ".join([f"{x:.2f}" for x in row]) for row in theta_img])
        self.lbl_theta.setText(f"Theta:\n{theta_str}")
        
        grid_vis = self.visualize_grid(grid_img)
        self.set_pixmap(self.lbl_grid, grid_vis)
        
    def visualize_grid(self, grid):
        h, w = grid.shape[:2]
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        step = max(1, h // 10)
        
        def get_pt(i, j):
            x = int((grid[i, j, 0] + 1) * w / 2)
            y = int((grid[i, j, 1] + 1) * h / 2)
            return (x, y)
            
        for i in range(0, h, step):
            for j in range(0, w, step):
                if j + step < w:
                    cv2.line(vis, get_pt(i, j), get_pt(i, j+step), (0, 255, 0), 1)
                if i + step < h:
                    cv2.line(vis, get_pt(i, j), get_pt(i+step, j), (0, 255, 0), 1)
        return vis
        
    def set_pixmap(self, label, img):
        if img.ndim == 2:
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format.Format_Grayscale8)
        else:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1], QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        
    def save_model(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Model", "stn_model.pth", "PyTorch Models (*.pth)")
        if path:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'in_channels': self.model.in_channels,
                'num_classes': self.model.fc2.out_features
            }, path)
            QMessageBox.information(self, "Success", f"Model saved to {path}")
            
    def export_logs(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Training Logs", "training_logs.csv", "CSV Files (*.csv)")
        if path:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Train Loss", "Train Acc", "Val Acc"])
                for i in range(self.log_table.rowCount()):
                    row_data = [self.log_table.item(i, j).text() if self.log_table.item(i, j) else "" for j in range(4)]
                    writer.writerow(row_data)
            QMessageBox.information(self, "Success", f"Logs exported to {path}")
            
    def show_model_summary(self):
        summary = f"Model Architecture:\n{self.model}\n\n"
        summary += f"Device: {self.device}\n"
        summary += f"Input Channels: {self.model.in_channels}\n"
        summary += f"Output Classes: {self.model.fc2.out_features}\n"
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        summary += f"Total Parameters: {total_params:,}\n"
        summary += f"Trainable Parameters: {trainable_params:,}"
        QMessageBox.information(self, "Model Summary", summary)
        
    def show_confusion_matrix(self):
        if self.test_loader is None:
            QMessageBox.warning(self, "Warning", "No validation set available!")
            return
            
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                output, _, _, _ = self.model(data)
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(target.numpy())
                
        num_classes = self.model.fc2.out_features
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(all_targets, all_preds):
            matrix[t, p] += 1
            
        dialog = ConfusionMatrixDialog(self)
        classes = [str(i) for i in range(num_classes)]
        dialog.plot_matrix(matrix, classes)
        dialog.exec()
        
    def train_finished(self):
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.btn_example_train.setEnabled(True)
        QMessageBox.information(self, "Success", "Training completed!")
        
    def train_error(self, msg):
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.btn_example_train.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)

# ==================== Inference Tab ====================
class InferenceTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.current_tensor = None
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        ctrl_layout = QHBoxLayout()
        self.btn_load_model = QPushButton("Load STN Model (.pth)")
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_load_image = QPushButton("Load Image")
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_load_video = QPushButton("Load Video")
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_start_cam = QPushButton("Start Webcam")
        self.btn_start_cam.clicked.connect(self.start_webcam)
        self.btn_stop_media = QPushButton("Stop Media")
        self.btn_stop_media.clicked.connect(self.stop_media)
        
        self.chk_manual_override = QCheckBox("Manual Theta Override")
        self.chk_manual_override.toggled.connect(self.toggle_manual_override)
        self.chk_saliency = QCheckBox("Show Saliency Map")
        self.chk_saliency.toggled.connect(self.run_inference)
        
        ctrl_layout.addWidget(self.btn_load_model)
        ctrl_layout.addWidget(self.btn_load_image)
        ctrl_layout.addWidget(self.btn_load_video)
        ctrl_layout.addWidget(self.btn_start_cam)
        ctrl_layout.addWidget(self.btn_stop_media)
        ctrl_layout.addWidget(self.chk_manual_override)
        ctrl_layout.addWidget(self.chk_saliency)
        layout.addLayout(ctrl_layout)
        
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        display_layout.setContentsMargins(0,0,0,0)
        self.lbl_original = self.create_img_label("Original Input\n(Drag & Drop Image Here)")
        self.lbl_transformed = self.create_img_label("STN Transformed")
        self.lbl_saliency = self.create_img_label("Saliency Map")
        display_layout.addWidget(self.lbl_original)
        display_layout.addWidget(self.lbl_transformed)
        display_layout.addWidget(self.lbl_saliency)
        main_splitter.addWidget(display_widget)
        
        theta_group = QGroupBox("Interactive Theta Matrix Control (Precision: 0.001)")
        theta_layout = QGridLayout()
        
        self.theta_sliders = {}
        params = [
            ("Scale X (a)", -2000, 2000, 1000, 0, 0),   
            ("Shear X (b)", -2000, 2000, 0, 0, 1),     
            ("Trans X (c)", -1000, 1000, 0, 0, 2),   
            ("Shear Y (d)", -2000, 2000, 0, 1, 0),     
            ("Scale Y (e)", -2000, 2000, 1000, 1, 1),   
            ("Trans Y (f)", -1000, 1000, 0, 1, 2)    
        ]
        
        for name, min_val, max_val, default, row, col in params:
            lbl_name = QLabel(name)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default)
            lbl_value = QLabel(f"{default/1000.0:.3f}")
            lbl_value.setMinimumWidth(50)
            slider.valueChanged.connect(self.apply_manual_theta)
            theta_layout.addWidget(lbl_name, row, col * 2)
            theta_layout.addWidget(slider, row, col * 2 + 1)
            theta_layout.addWidget(lbl_value, row, col * 2 + 2)
            self.theta_sliders[f"{row}{col}"] = (slider, lbl_value)
            
        self.btn_reset_theta = QPushButton("Reset to Identity")
        self.btn_reset_theta.clicked.connect(self.reset_theta)
        theta_layout.addWidget(self.btn_reset_theta, 2, 0, 1, 6)
        
        theta_group.setLayout(theta_layout)
        main_splitter.addWidget(theta_group)
        
        layout.addWidget(main_splitter)
        self.setLayout(layout)
        
    def create_img_label(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("background-color: #2b2b2b; color: #888; border: 1px solid #555;")
        lbl.setMinimumSize(300, 300)
        return lbl
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.stop_media()
                self.load_image(path=path)
                
    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load STN Model", "", "PyTorch Models (*.pth)")
        if path:
            try:
                checkpoint = torch.load(path, map_location=self.device)
                in_ch = checkpoint.get('in_channels', 1)
                num_cls = checkpoint.get('num_classes', 10)
                self.model = STNModel(in_channels=in_ch, num_classes=num_cls).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                QMessageBox.information(self, "Success", f"Model loaded: {in_ch} channels, {num_cls} classes.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                
    def load_image(self, path=None):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load a model first!")
            return
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if path:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.model.in_channels == 1 else cv2.IMREAD_COLOR)
            if img is None: return
            
            img_resized = cv2.resize(img, (32, 32))
            tensor = torch.from_numpy(img_resized).float() / 255.0
            if self.model.in_channels == 1:
                tensor = tensor.unsqueeze(0).unsqueeze(0) 
            else:
                tensor = tensor.permute(2, 0, 1).unsqueeze(0) 
            tensor = (tensor - 0.5) / 0.5 
            self.current_tensor = tensor.to(self.device)
            
            self.run_inference()
            
    def load_video(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load a model first!")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.stop_media()
            self.cap = cv2.VideoCapture(path)
            self.timer.start(30) # ~33 fps
            
    def start_webcam(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load a model first!")
            return
        self.stop_media()
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        
    def stop_media(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.stop_media()
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.stop_media()
            return
            
        # Preprocess frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if self.model.in_channels == 1 else frame
        img_resized = cv2.resize(img, (32, 32))
        tensor = torch.from_numpy(img_resized).float() / 255.0
        if self.model.in_channels == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0) 
        else:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0) 
        tensor = (tensor - 0.5) / 0.5 
        self.current_tensor = tensor.to(self.device)
        
        self.run_inference()
        
    def run_inference(self):
        if self.current_tensor is None or self.model is None: return
        
        with torch.no_grad():
            if self.chk_manual_override.isChecked():
                self.apply_manual_theta()
            else:
                _, transformed, theta, _ = self.model(self.current_tensor)
                self.display_results(self.current_tensor, transformed, theta.cpu().numpy())
                
        if self.chk_saliency.isChecked():
            self.compute_saliency()
        else:
            self.lbl_saliency.clear()
            self.lbl_saliency.setText("Saliency Map")
            
    def compute_saliency(self):
        if self.current_tensor is None or self.model is None: return
        
        # Need gradients for input
        self.current_tensor.requires_grad_(True)
        self.model.eval()
        output, _, _, _ = self.model(self.current_tensor)
        target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        saliency = self.current_tensor.grad.data.abs().squeeze()
        if saliency.ndim == 3:
            saliency = torch.max(saliency, dim=0)[0]
            
        saliency = saliency.cpu().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        saliency = (saliency * 255).astype(np.uint8)
        saliency = cv2.resize(saliency, (300, 300))
        saliency = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
        
        self.set_pixmap(self.lbl_saliency, saliency, is_gray=False)
        
        # Detach tensor for future use
        self.current_tensor = self.current_tensor.detach()
        
    def apply_manual_theta(self):
        if self.current_tensor is None or self.model is None: return
        
        a = self.theta_sliders["00"][0].value() / 1000.0
        b = self.theta_sliders["01"][0].value() / 1000.0
        c = self.theta_sliders["02"][0].value() / 1000.0
        d = self.theta_sliders["10"][0].value() / 1000.0
        e = self.theta_sliders["11"][0].value() / 1000.0
        f = self.theta_sliders["12"][0].value() / 1000.0
        
        self.theta_sliders["00"][1].setText(f"{a:.3f}")
        self.theta_sliders["01"][1].setText(f"{b:.3f}")
        self.theta_sliders["02"][1].setText(f"{c:.3f}")
        self.theta_sliders["10"][1].setText(f"{d:.3f}")
        self.theta_sliders["11"][1].setText(f"{e:.3f}")
        self.theta_sliders["12"][1].setText(f"{f:.3f}")
        
        theta = torch.tensor([[[a, b, c], [d, e, f]]], dtype=torch.float32).to(self.device)
        grid = F.affine_grid(theta, self.current_tensor.size(), align_corners=True)
        transformed = F.grid_sample(self.current_tensor, grid, align_corners=True)
        
        self.display_results(self.current_tensor, transformed, theta.cpu().numpy())
        
    def reset_theta(self):
        self.theta_sliders["00"][0].setValue(1000) 
        self.theta_sliders["01"][0].setValue(0)   
        self.theta_sliders["02"][0].setValue(0)   
        self.theta_sliders["10"][0].setValue(0)   
        self.theta_sliders["11"][0].setValue(1000) 
        self.theta_sliders["12"][0].setValue(0)   
        
    def toggle_manual_override(self, checked):
        self.run_inference()
        
    def display_results(self, original_t, transformed_t, theta):
        orig_img = original_t[0].cpu().numpy().squeeze()
        if orig_img.ndim == 3: orig_img = np.transpose(orig_img, (1, 2, 0))
        trans_img = transformed_t[0].cpu().numpy().squeeze()
        if trans_img.ndim == 3: trans_img = np.transpose(trans_img, (1, 2, 0))
        
        self.set_pixmap(self.lbl_original, orig_img, self.model.in_channels==1)
        self.set_pixmap(self.lbl_transformed, trans_img, self.model.in_channels==1)
        
        theta_str = "\n".join(["  ".join([f"{x:.3f}" for x in row]) for row in theta[0]])
        self.lbl_transformed.setToolTip(f"Theta Matrix:\n{theta_str}")
        
    def set_pixmap(self, label, img, is_gray=False):
        if is_gray:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img
            img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format.Format_Grayscale8)
        else:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img
            img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[2] * img.shape[1], QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

# ==================== Main Window ====================
class SpatialTransformerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spatial Transformer Studio Pro")
        self.resize(1600, 1000)
        
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        self.manual_tab = ManualTransformTab()
        self.training_tab = TrainingTab()
        self.inference_tab = InferenceTab()
        
        self.tab_widget.addTab(self.manual_tab, "Manual Dataset Generator")
        self.tab_widget.addTab(self.training_tab, "STN Training")
        self.tab_widget.addTab(self.inference_tab, "STN Inference & Video")
        
        # GPU VRAM Timer
        self.vram_timer = QTimer(self)
        self.vram_timer.timeout.connect(self.update_status_bar)
        self.vram_timer.start(1000) # Update every 1 second
        self.update_status_bar()
        
    def update_status_bar(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            cached = torch.cuda.memory_reserved() / (1024 ** 2)
            self.statusBar().showMessage(f"Device: {torch.cuda.get_device_name(0)} | VRAM Allocated: {allocated:.1f} MB | Cached: {cached:.1f} MB")
        else:
            self.statusBar().showMessage("Device: CPU")
        
    def closeEvent(self, event):
        if hasattr(self.training_tab, 'training_thread') and self.training_tab.training_thread:
            self.training_tab.training_thread.stop()
        if hasattr(self.inference_tab, 'timer'):
            self.inference_tab.stop_media()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = SpatialTransformerApp()
    window.show()
    sys.exit(app.exec())