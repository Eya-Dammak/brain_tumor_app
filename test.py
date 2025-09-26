import sys
import os
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog,
    QTextEdit, QLineEdit, QTabWidget, QFormLayout
)
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy,QFrame
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QScrollArea, QStackedLayout, QFileDialog, QFormLayout,
    QGraphicsOpacityEffect
)
from PyQt5.QtGui import QPalette, QLinearGradient, QColor, QBrush
from PyQt5.QtCore import QPropertyAnimation, QTimer, QPointF, QEasingCurve
# ------------------ Classification Model ------------------
classification_model = load_model("classification_model.keras")
class_mappings = ['Glioma', 'Meninigioma', 'Notumor', 'Pituitary']

def preprocess_classification(img_path):
    img = Image.open(img_path).convert('L').resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def classify_image(img_path):
    img = preprocess_classification(img_path)
    prediction = classification_model.predict(img)
    idx = np.argmax(prediction)
    return class_mappings[idx], float(prediction[0][idx])

# ------------------ U-Net Segmentation Model ------------------
def conv_block(inputs, filters):
    x = Conv2D(filters, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def encoder_block(inputs, filters):
    x = conv_block(inputs, filters)
    p = MaxPooling2D(2)(x)
    return x, p

def decoder_block(inputs, filters, skip):
    x = Conv2DTranspose(filters, 2, strides=2, padding='same')(inputs)
    x = concatenate([x, skip])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)
    b1 = conv_block(p4, 512)
    d1 = decoder_block(b1, 256, s4)
    d2 = decoder_block(d1, 128, s3)
    d3 = decoder_block(d2, 64, s2)
    d4 = decoder_block(d3, 32, s1)
    outputs = Conv2D(1, 1, activation="sigmoid")(d4)
    return Model(inputs, outputs)

segmentation_model = build_unet()
segmentation_model.load_weights("segmentation_model.hdf5")

def segment_and_highlight(image_path, output_path="highlighted_output.jpg"):
    # Step 1: Load and preprocess the image
    img = Image.open(image_path).convert('L').resize((256, 256))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))

    # Step 2: Predict mask
    mask = segmentation_model.predict(arr)[0, :, :, 0]

    # DEBUG: Save raw mask and print mask stats
    print(f"Mask max value: {np.max(mask):.4f}")
    Image.fromarray((mask * 255).astype(np.uint8)).save("raw_mask.jpg")

    # Step 3: Binarize the mask with a lower threshold
    threshold = 0.3  # Increased sensitivity
    binary_mask = (mask > threshold).astype(np.uint8)
    print(f"Sum of binary mask: {np.sum(binary_mask)}")  # Print how many pixels were marked

    # Step 4: Overlay red on tumor area
    original = Image.open(image_path).convert("RGB").resize((256, 256))
    original_arr = np.array(original)

    # Strong red overlay (not semi-transparent)
    highlighted = original_arr.copy()
    red_color = np.array([255, 0, 0], dtype=np.uint8)
    highlighted[binary_mask == 1] = (
        0.5 * red_color + 0.5 * original_arr[binary_mask == 1]
    ).astype(np.uint8)

    # Step 5: Save and return
    Image.fromarray(highlighted).save(output_path)
    return output_path
class ImageUploadWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.layout = QVBoxLayout()
        self.image_path = None
        self.scale_factor = 1.0  # ← Added zoom scale

            # === Affichage côte à côte : image originale à gauche, segmentée à droite ===
        self.image_layout = QHBoxLayout()

        self.label = QLabel("Drag & Drop or Click to Upload MRI Image")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(
            "border: 2px dashed #95a5a6; padding: 40px; background-color: #ecf0f1; "
            "font-size: 16px; color: #2c3e50;"
        )
        self.image_layout.addWidget(self.label)

        # === Bloc image segmentée avec titre ===
        self.segmented_layout = QVBoxLayout()

        self.segmented_title = QLabel("🟥 Tumor highlighted in image below.")
        self.segmented_title.setAlignment(Qt.AlignCenter)
        self.segmented_title.setStyleSheet("font-size: 14px; color: #c0392b;")
        self.segmented_title.setVisible(False)  # Affiché uniquement après prédiction

        self.highlighted_img = QLabel()
        self.highlighted_img.setAlignment(Qt.AlignCenter)
        self.highlighted_img.setStyleSheet(
            "border: 2px dashed #95a5a6; padding: 10px; background-color: #ecf0f1;"
        )

        self.segmented_layout.addWidget(self.segmented_title)
        self.segmented_layout.addWidget(self.highlighted_img)

        self.image_layout.addLayout(self.segmented_layout)


        self.layout.addLayout(self.image_layout)

        # === Bouton pour parcourir ===
        self.button = QPushButton("📁 Browse Image")
        self.button.setStyleSheet(
            "background-color: #3498db; color: white; font-size: 16px; padding: 10px; border-radius: 8px;"
        )
        self.button.clicked.connect(self.open_file_dialog)
        self.layout.addWidget(self.button)


        # Zoom Buttons
        self.zoom_layout = QHBoxLayout()
        self.zoom_in_button = QPushButton("🔍➕ Zoom In")
        self.zoom_out_button = QPushButton("🔍➖ Zoom Out")
        for btn in (self.zoom_in_button, self.zoom_out_button):
            btn.setVisible(False)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f39c12;
                    color: white;
                    font-size: 14px;
                    padding: 6px 12px;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #d68910;
                }
            """)
            self.zoom_layout.addWidget(btn)
            self.layout.addLayout(self.zoom_layout)  # ← Fixed here

            self.predict_button = QPushButton("🧠 Predict")
            self.predict_button.setStyleSheet("background-color: #27ae60; color: white; font-size: 16px; padding: 10px; border-radius: 8px;")
            self.predict_button.setVisible(False)
            self.predict_button.clicked.connect(self.run_prediction)
            self.layout.addWidget(self.predict_button)

            self.result_label = QLabel()
            self.result_label.setStyleSheet("font-size: 15px; color: #2c3e50; padding: 10px;")
            self.result_label.setWordWrap(True)
            self.layout.addWidget(self.result_label)

            self.setLayout(self.layout)

            # Zoom Connections
            self.zoom_in_button.clicked.connect(self.zoom_in)
            self.zoom_out_button.clicked.connect(self.zoom_out)

    def zoom_in(self):
        self.scale_factor *= 1.2
        self.update_image_display()

    def zoom_out(self):
        self.scale_factor /= 1.2
        self.update_image_display()

    def open_file_dialog(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file:
            self.image_path = file
            self.scale_factor = 1.0  # Reset zoom
            self.update_image_display()
            self.predict_button.setVisible(True)

    def show_image(self, path):
        self.image_path = path
        self.scale_factor = 1.0  # Reset zoom
        self.update_image_display()

    def update_image_display(self):
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            width = int(300 * self.scale_factor)
            scaled = pixmap.scaledToWidth(width, Qt.SmoothTransformation)
            self.label.setPixmap(scaled)
            self.zoom_in_button.setVisible(True)
            self.zoom_out_button.setVisible(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            self.show_image(path)
            self.predict_button.setVisible(True)

    def run_prediction(self):
        if not self.image_path:
            return

        tumor_class, prob = classify_image(self.image_path)
        result_text = f"🧠 Prediction: <b>{tumor_class}</b><br>🎯 Probability: {prob:.2%}"
        self.result_label.setText(result_text)

        # Load original image (left side)
        pixmap_original = QPixmap(self.image_path).scaledToWidth(300, Qt.SmoothTransformation)
        self.label.setPixmap(pixmap_original)

        if tumor_class != "Notumor":
            highlighted_path = segment_and_highlight(self.image_path)
            pixmap_highlighted = QPixmap(highlighted_path).scaledToWidth(300, Qt.SmoothTransformation)
            self.highlighted_img.setPixmap(pixmap_highlighted)
            self.segmented_title.setVisible(True)  # Affiche le titre
        else:
            self.highlighted_img.clear()
            self.segmented_title.setVisible(False)


from chatbot_engine import ChatbotAssistant  # ← Assure-toi que chatbot_engine.py est dans le même dossier

class ContactForm(QWidget):
    def __init__(self):
        super().__init__()

        self.assistant = ChatbotAssistant("intents.json")
        self.assistant.load_model("chatbot_model.pth", "dimensions.json")

        layout = QVBoxLayout()

        # Zone d'affichage du chat
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            background-color: #ffffff;
            padding: 15px;
            border: 1px solid #ccc;
            font-size: 16px;
            font-family: 'Segoe UI', sans-serif;
            border-radius: 12px;
        """)
        layout.addWidget(self.chat_display)

        # 🟪 Boutons Help / Services / Clear (au-dessus de l'input)
        top_buttons = QHBoxLayout()
        help_btn = QPushButton("🆘 Help")
        services_btn = QPushButton("🛠 Services")
        clear_btn = QPushButton("🧼 Clear")
        for btn in (help_btn, services_btn, clear_btn):
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #bdc3c7;
                    border-radius: 12px;
                    font-size: 14px;
                    padding: 8px 16px;
                    font-family: 'Segoe UI';
                }
                QPushButton:hover {
                    background-color: #95a5a6;
                }
            """)
            top_buttons.addWidget(btn)
        layout.addLayout(top_buttons)

        # Champs d’entrée + bouton envoyer
        input_layout = QHBoxLayout()
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your question here...")
        self.user_input.setStyleSheet("""
            padding: 12px;
            font-size: 16px;
            font-family: 'Segoe UI', sans-serif;
            border-radius: 20px;
            border: 1px solid #aaa;
            background-color: #ecf0f1;
        """)
        send_btn = QPushButton("➤")
        send_btn.setStyleSheet("""
            background-color: #3498db;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 20px;
        """)
        input_layout.addWidget(self.user_input)
        input_layout.addWidget(send_btn)
        layout.addLayout(input_layout)

        # Connexions (modifiées)
        send_btn.clicked.connect(self.send_message)
        self.user_input.returnPressed.connect(self.send_message)
        help_btn.clicked.connect(lambda: self.send_message("help"))
        services_btn.clicked.connect(lambda: self.send_message("services"))
        clear_btn.clicked.connect(self.chat_display.clear)

        self.setLayout(layout)

    def display_message(self, text, sender="user"):
        if sender == "user":
            bg_color = "#a1d0f0"
            text_color = "black"
            align = "right"
            radius = "20px 20px 0px 20px"
        else:
            bg_color = "#d1d1d1"
            text_color = "black"
            align = "left"
            radius = "20px 20px 0px 20px"

        html = f"""
        <div style='text-align: {align}; margin: 10px 0;'>
            <span style='
                display: inline-block;
                background-color: {bg_color};
                color: {text_color};
                padding: 14px 18px;
                font-size: 16px;
                font-family: "Segoe UI", sans-serif;
                border-radius: {radius};
                max-width: 70%;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
            '>
                {text}
            </span>
        </div>
        """
        self.chat_display.append(html)

    def display_bot_message(self, text):
        self.display_message(text, sender="bot")

    # send_message avec argument optionnel 'predefined_text'
    def send_message(self, predefined_text=None):
        if predefined_text is None:
            message = self.user_input.text().strip()
        else:
            message = predefined_text

        if message:
            self.display_message(message, sender="user")
            response = self.assistant.process_message(message)
            self.display_bot_message(response)
            if predefined_text is None:
                self.user_input.clear()


class TeamSection(QWidget):
    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout()
        # === Intro paragraph in a single styled box ===
        intro_text = QLabel(
            "This desktop app, created by Eya Damak and Chahd Gharbi from Sfax Engineering School, "
            "helps doctors quickly detect brain tumors by simplifying MRI image analysis, improving diagnosis speed and accuracy. "
            "The project was supervised by Assistant Professor Olfa Gaddour."
        )
        intro_text.setWordWrap(True)
        intro_text.setAlignment(Qt.AlignCenter)
        intro_text.setStyleSheet("""
            background-color: #ecf0f1;
            border: 2px solid #bdc3c7;
            border-radius: 10px;
            padding: 15px;
            font-size: 14px;
            color: #2c3e50;
        """)

        main_layout.addWidget(intro_text)


        # === SINGLE ROW FOR ALL 3 CARDS ===
        row_layout = QHBoxLayout()

        # OLFA card (left)
        olfa_card = self.create_member_card(
            name="Olfa Gaddour",
            role="Assistant Professor",
            img_path="C:/Users/GAMING/Desktop/medapp/olfa.jpg",
            linkedin_url="https://www.linkedin.com/in/olfa-gaddour-7724375/",
            facebook_url="https://www.facebook.com/olfa.gaddour?locale=fr_FR"
        )
        row_layout.addWidget(olfa_card)

        # EYA card (middle)
        eya_card = self.create_member_card(
            name="Eya Damak",
            role="First-year CS Student",
            img_path="C:/Users/GAMING/Desktop/medapp/eya.jpg",
            linkedin_url="https://www.linkedin.com/in/eya-dammak-498347331/",
            facebook_url="https://www.facebook.com/eya.dammak.965?locale=fr_FR"
        )
        row_layout.addWidget(eya_card)

        # CHAHD card (right)
        chahd_card = self.create_member_card(
            name="Chahd Gharbi",
            role="First-year CS Student",
            img_path="C:/Users/GAMING/Desktop/medapp/chahd.jpg",
            linkedin_url="https://www.linkedin.com/in/chahd-gharbi-b96345331/",
            facebook_url="https://www.facebook.com/chahd.gharbi.9?locale=fr_FR"
        )
        row_layout.addWidget(chahd_card)

        # Set layout
        main_layout.addLayout(row_layout)
        self.setLayout(main_layout)
    
    def create_member_card(self, name, role, img_path, linkedin_url="", facebook_url=""):
        card = QWidget()
        layout = QVBoxLayout()
        card.setStyleSheet("""
            background-color: #e0dede;
            border-radius: 10px;
            padding: 15px;
        """)

        # Profile image
        pixmap = QPixmap(img_path).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img_label = QLabel()
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignCenter)

        # Name and role
        name_label = QLabel(f"<b>{name}</b>")
        name_label.setAlignment(Qt.AlignCenter)

        role_label = QLabel(role)
        role_label.setAlignment(Qt.AlignCenter)
        role_label.setStyleSheet("color: #2c3e50; font-size: 13px;")

        # Social media links with icons
        social_layout = QHBoxLayout()
        social_layout.setAlignment(Qt.AlignCenter)

        if facebook_url:
            fcb_label = QLabel(f"""
                <a href="{facebook_url}">
                    <img src="C:/Users/GAMING/Desktop/medapp/facebook.png" width="24" height="24">
                </a>
            """)
            fcb_label.setOpenExternalLinks(True)
            social_layout.addWidget(fcb_label)

        if linkedin_url:
            linkedin_label = QLabel(f"""
                <a href="{linkedin_url}">
                    <img src="C:/Users/GAMING/Desktop/medapp/linkedin.png" width="24" height="24">
                </a>
            """)
            linkedin_label.setOpenExternalLinks(True)
            social_layout.addWidget(linkedin_label)

        # Add widgets to layout
        layout.addWidget(img_label)
        layout.addWidget(name_label)
        layout.addWidget(role_label)
        layout.addLayout(social_layout)
        card.setLayout(layout)
        card.setFixedWidth(250)

        return card
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Brain Tumor MRI Analyzer")
        self.setStyleSheet("font-family: Arial;")
        self.setAutoFillBackground(True)
        self.update_gradient_background()

        self.main_layout = QVBoxLayout(self)
        self.stack = QStackedLayout()
        self.main_layout.addLayout(self.stack)

        self.home_widget = self.build_home()
        self.stack.addWidget(self.home_widget)

        self.menu_widget = self.build_menu()
        self.stack.addWidget(self.menu_widget)

        self.sections = {
            "predict": self.create_section("🧪 Predict", ImageUploadWidget()),
            "services": self.create_section("🛠 Services", self.build_services_section()),
            "performance": self.create_section("📊 Performance", self.build_performance_section()),
            "contact": self.create_section("✉️ Contact", ContactForm()),
            "team": self.create_section("👥 Team", TeamSection())
        }

        for widget in self.sections.values():
            self.stack.addWidget(widget)

        self.start_gradient_animation()

    def update_gradient_background(self, shift=0):
        gradient = QLinearGradient(QPointF(shift, 0), QPointF(1 + shift, 1))
        gradient.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
        gradient.setColorAt(0.0, QColor("#f7b16e"))
        gradient.setColorAt(0.5, QColor("#f9c88c"))
        gradient.setColorAt(1.0, QColor("#fae1bf"))
        palette = self.palette()
        palette.setBrush(self.backgroundRole(), QBrush(gradient))
        self.setPalette(palette)

    def start_gradient_animation(self):
        self.gradient_shift = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_gradient)
        self.timer.start(30)

    def animate_gradient(self):
        self.gradient_shift += 0.015
        if self.gradient_shift > 1:
            self.gradient_shift = 0
        self.update_gradient_background(self.gradient_shift)

    def build_home(self):
        w = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)

        self.title = QLabel("<h1 style='color:#2c3e50;'>MRI Images Brain Tumor Classification & Segmentation</h1>")
        self.title.setAlignment(Qt.AlignCenter)

        self.subtitle = QLabel("🎯 An Innovative Solution to Facilitate Brain Tumor Detection Through MRI Image Uploads!")
        self.subtitle.setStyleSheet("font-size: 16px; color: #34495e;")
        self.subtitle.setAlignment(Qt.AlignCenter)

        self.get_started = QPushButton("🚀 Get Started")
        self.get_started.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                color: white;
                font-size: 16px;
                padding: 10px 20px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
        """)
        self.get_started.clicked.connect(self.go_to_menu)

        layout.addStretch()
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addStretch()
        layout.addWidget(self.get_started, alignment=Qt.AlignCenter)

        self.fade_in(self.title)
        self.fade_in(self.subtitle)
        self.fade_in(self.get_started)

        w.setLayout(layout)
        return w

    def build_menu(self):
        w = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(100, 50, 100, 50)
        layout.setSpacing(20)

        sections = [
            ("🧪 Predict", "predict"),
            ("🛠 Services", "services"),
            ("📊 Performance", "performance"),
            ("✉️ Contact", "contact"),
            ("👥 Team", "team"),
        ]

        for title, key in sections:
            btn = QPushButton(title)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #e67e22;
                    color: white;
                    font-size: 18px;
                    padding: 15px;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: #d35400;
                }
            """)
            btn.clicked.connect(lambda _, k=key: self.go_to_section(k))
            layout.addWidget(btn)

        w.setLayout(layout)
        return w

    def create_section(self, title_text, content_widget):
        w = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        back_btn = QPushButton("← Return")
        back_btn.setFixedWidth(100)
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #2c3e50;
                font-size: 16px;
                border: none;
            }
            QPushButton:hover {
                color: #e67e22;
            }
        """)
        back_btn.clicked.connect(self.go_to_menu)

        title = QLabel(f"<h2>{title_text}</h2>")
        title.setStyleSheet("color: #2c3e50;")
        title.setAlignment(Qt.AlignCenter)

        layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        layout.addWidget(title)
        layout.addWidget(content_widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container.setLayout(layout)
        scroll.setWidget(container)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        w.setLayout(main_layout)
        return w

    
    def build_services_section(self):
        w = QWidget()
        layout = QVBoxLayout()

        services = [
            ("🧠 MRI Image Classification", "Upload MRI brain images and instantly receive an accurate diagnosis indicating the presence or absence of a brain tumor using advanced deep learning models."),
            ("📨 Diagnostic Results Messaging", "Get clear and immediate feedback with detailed messages about your MRI scan results, helping doctors make informed decisions quickly and efficiently."),
            ("🎯 Tumor Segmentation and Highlighting", "If a tumor is detected, easily perform image segmentation to visually highlight tumor areas on the MRI scans, aiding precise analysis and treatment planning."),
        ]

        for title, description in services:
            card = QWidget()
            card_layout = QVBoxLayout()
            card.setStyleSheet("""
                background-color:#e0dede;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 10px;
            """)

            title_label = QLabel(f"<b>{title}</b>")
            desc_label = QLabel(description)
            desc_label.setStyleSheet("font-size: 13px; color: #2c3e50;")

            card_layout.addWidget(title_label)
            card_layout.addWidget(desc_label)
            card.setLayout(card_layout)

            layout.addWidget(card)

        layout.addStretch()
        w.setLayout(layout)
        return w

    def build_performance_section(self):
        w = QWidget()
        layout = QVBoxLayout()
        metrics = {
            "Accuracy": "98.7%",
            "Recall": "99%",
            "F1-Score": "99%",
            "Precision": "98.5%",
        }

        for key, value in metrics.items():
            label = QLabel(f"<b>{key}:</b> {value}")
            label.setStyleSheet(
                """
                font-size: 16px;
                color: black;
                padding: 10px;
                background-color: #D3D3D3;  /* light grey */
                border-radius: 8px;
                margin-bottom: 8px;
                """
            )
            layout.addWidget(label)

        w.setLayout(layout)
        return w

        
    def fade_in(self, widget):
        effect = QGraphicsOpacityEffect()
        widget.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(1000)
        animation.setStartValue(0)
        animation.setEndValue(1)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        animation.start()
        setattr(widget, "_fade_anim", animation)

    def go_to_menu(self):
        self.stack.setCurrentWidget(self.menu_widget)

    def go_to_section(self, key):
        widget = self.sections.get(key)
        if widget:
            self.stack.setCurrentWidget(widget)

# === 5. Run App ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(900, 600)
    window.show()
    sys.exit(app.exec_())