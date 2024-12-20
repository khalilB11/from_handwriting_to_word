from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QWidget, QTableWidget,
                             QTableWidgetItem, QFileDialog, QFrame, QTextEdit, QGraphicsView, QGraphicsScene,
                             QGraphicsPixmapItem, QProgressBar, QListWidgetItem, QListWidget)
from PyQt5.QtCore import Qt, QMimeData, QRectF, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QPainter, QIcon
from PIL import Image
import sys
import os
import numpy as np
from scipy.signal import savgol_filter
import cv2
from collections import Counter
from scipy.ndimage import convolve1d, gaussian_filter1d
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from torchvision import transforms
import cv2
import pytesseract
from PIL import Image
import numpy as np


class InferenceWorker(QThread):
    progress = pyqtSignal(int)  # Signal for progress updates
    finished = pyqtSignal(list)  # Signal when inference is finished
    error = pyqtSignal(str)  # Signal for errors

    def __init__(self, model, processor, images):
        super().__init__()
        self.model = model
        self.processor = processor
        self.images = images

    def model_inference(self, images):
        list_word = []
        for word_image in images:
            image_array = (word_image * 255).astype(np.uint8)
            _word_image = Image.fromarray(image_array).convert("RGB")
            pixel_values = self.processor(_word_image, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            list_word.append(generated_text)
        return list_word

    def run(self):
        try:
            num_images = len(self.images)
            results = []
            for idx, image in enumerate(self.images):
                word = self.model_inference([image])
                results.append(word)
                progress = int((idx + 1) / num_images * 100)
                self.progress.emit(progress)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class ProductPrototypeTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Product Prototype Tool")
        self.setGeometry(50, 50, 1800, 980)

        # Main container widget
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        # Layout for the main widget
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)

        # Column 1: Scanned Documents
        self.column1 = QVBoxLayout()
        self.column1_frame = QFrame()
        self.column1_frame.setLayout(self.column1)
        self.column1_frame.setFrameShape(QFrame.Box)
        self.column1_frame.setFrameShadow(QFrame.Raised)

        self.drop_label = QLabel("Drag and drop scanned documents here")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet("border: 1px dashed gray;")
        self.drop_label.setAcceptDrops(True)
        self.drop_label.dragEnterEvent = self.drag_enter_event
        self.drop_label.dropEvent = self.drop_event
        self.column1.addWidget(self.drop_label)

        self.browse_button = QPushButton("Browse File")
        self.browse_button.clicked.connect(self.browse_file)
        self.column1.addWidget(self.browse_button)

        self.image_view = QGraphicsView()
        self.image_scene = QGraphicsScene()
        self.image_view.setScene(self.image_scene)
        self.column1.addWidget(self.image_view)

        self.main_layout.addWidget(self.column1_frame, 2)

        # Column 2: Text Display
        self.column2 = QVBoxLayout()
        self.column2_frame = QFrame()
        self.column2_frame.setLayout(self.column2)
        self.column2_frame.setFrameShape(QFrame.Box)
        self.column2_frame.setFrameShadow(QFrame.Raised)

        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.column2.addWidget(self.text_display)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.column2.addWidget(self.progress_bar)

        self.main_layout.addWidget(self.column2_frame, 2)

        # Column 3: Vertical List of Images and Table
        self.column3 = QVBoxLayout()
        self.column3_frame = QFrame()
        self.column3_frame.setLayout(self.column3)
        self.column3_frame.setFrameShape(QFrame.Box)
        self.column3_frame.setFrameShadow(QFrame.Raised)

        # Image list widget for vertical layout
        self.image_list_widget = QListWidget()
        self.image_list_widget.setViewMode(QListWidget.IconMode)
        self.image_list_widget.setFlow(QListWidget.TopToBottom)  # Arrange items vertically
        self.image_list_widget.setSpacing(10)  # Add spacing between items
        self.image_list_widget.setResizeMode(QListWidget.Adjust)  # Dynamically adjust sizes
        self.image_list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Enable vertical scrolling

        # Add the image list widget to column 3
        self.column3.addWidget(self.image_list_widget)

        # Table for features
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["Feature", "Value", "Description"])
        self.column3.addWidget(self.table_widget)

        # Footer layout for table actions
        self.footer_layout = QHBoxLayout()
        self.add_row_button = QPushButton("Add Row")
        self.add_row_button.clicked.connect(self.add_row)
        self.footer_layout.addWidget(self.add_row_button)

        self.remove_row_button = QPushButton("Remove Selected Row")
        self.remove_row_button.clicked.connect(self.remove_selected_row)
        self.footer_layout.addWidget(self.remove_row_button)

        self.column3.addLayout(self.footer_layout)

        # Add the column 3 frame to the main layout
        self.main_layout.addWidget(self.column3_frame, 1)

        # Add images to the vertical list
        self.add_images_to_list([
            "word_template_1.png",
            "word_template_1.png",
            "word_template_1.png"
        ])
        self.main_layout.addWidget(self.column3_frame, 1)

        # Load the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    def add_images_to_list(self, images):
        self.image_list_widget.clear()  # Clear existing items
        img_dir = 'word_templates'
        for image_name in images:
            image_path = os.path.join(img_dir, image_name)
            item = QListWidgetItem()
            pixmap = QPixmap(image_path).scaled(
                self.image_list_widget.iconSize(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            item.setIcon(QIcon(pixmap))
            item.setText(os.path.basename(image_path))
            self.image_list_widget.addItem(item)

    def add_images_to_list(self, images):
        """Add images to the vertical list."""
        self.image_list_widget.clear()  # Clear existing items
        img_dir = 'word_templates'
        for image_name in images:
            image_path = os.path.join(img_dir, image_name)
            item = QListWidgetItem()

            # Load and resize the image to fit the column width
            pixmap = QPixmap(image_path)
            column_width = self.column3_frame.width() - 20  # Leave some padding
            scaled_pixmap = pixmap.scaled(column_width, column_width, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            item.setIcon(QIcon(scaled_pixmap))
            item.setText(os.path.basename(image_path))
            item.setSizeHint(QSize(column_width, scaled_pixmap.height() + 20))  # Adjust item size to fit
            self.image_list_widget.addItem(item)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        column_width = self.column3_frame.width()
        icon_size = QSize(column_width - 40, 150)  # Adjust height as needed
        self.image_list_widget.setIconSize(icon_size)
        self.image_list_widget.setGridSize(icon_size + QSize(20, 20))

    def drag_enter_event(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def drop_event(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(".png"):
                self.load_and_process_image(file_path)

    def browse_file(self):
        # Open file dialog starting from the current directory
        current_dir = os.getcwd()  # Get the current working directory
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PNG File", current_dir, "Image Files (*.png)")
        if file_path:
            self.load_and_process_image(file_path)

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image not found: " + image_path)
        image = cv2.resize(image, (512, 512))
        image = image / 255.0

        # Detect text
        result = self.detect_90_degree_corners_and_text(image_path)
        if result['box_details']:
            y_answer_pos = [(pos['corners'][0][0], pos['corners'][3][0]) for pos in result['box_details']]
        return image, y_answer_pos

    def split_into_lines(self, image):
        height, width = image.shape
        box_height = 5
        step = 2
        all_steps_inf_content = []
        positions = []

        for pos in range(0, height - box_height + 1, step):
            image_seg = image[pos:pos + box_height, :]
            sum_info_content = np.sum(image_seg)
            all_steps_inf_content.append(sum_info_content)
            positions.append(pos)

        np_positions = np.array(positions)
        np_all_steps_inf_content = np.array(all_steps_inf_content)

        sm_np_all_steps_inf_content = savgol_filter(np_all_steps_inf_content, window_length=11, polyorder=3)
        maxima_indices = (np.diff(np.sign(np.diff(sm_np_all_steps_inf_content))) < 0).nonzero()[0] + 1
        separ_lines = [np_positions[idx] for idx in maxima_indices]

        lines = []
        for idx, pos in enumerate(separ_lines):
            if idx < len(separ_lines) - 1:
                line = image[pos:separ_lines[idx + 1], :]
                lines.append(line)
            else:
                line = image[pos:, :]
                lines.append(line)

        return lines, image, separ_lines

    def load_and_process_image(self, file_path):
        try:
            image, answer_box = self.preprocess_image(file_path)
            if image is None:
                raise ValueError("Image not found: " + file_path)

            line_images, processed_image, separ_lines = self.split_into_lines(image)

            height, width = processed_image.shape
            pixmap = QPixmap(file_path).scaled(self.column1_frame.width(), height, Qt.KeepAspectRatio)

            self.image_scene.clear()
            self.image_scene.addPixmap(pixmap)
            self.image_view.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))

            self.progress_bar.setValue(0)
            self.worker = InferenceWorker(self.model, self.processor, line_images)
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.finished.connect(self.on_inference_finished)
            self.worker.error.connect(self.on_inference_error)
            self.worker.start()
        except Exception as e:
            self.text_display.setText(f"Failed to process image: {str(e)}")

    def on_inference_finished(self, results):
        text = "\n".join(" ".join(line) for line in results)
        self.text_display.setText(text)

    def on_inference_error(self, error_message):
        self.text_display.setText(f"Error during inference: {error_message}")

    def add_row(self):
        row_position = self.table_widget.rowCount()
        self.table_widget.insertRow(row_position)

    def remove_selected_row(self):
        selected_row = self.table_widget.currentRow()
        if selected_row != -1:
            self.table_widget.removeRow(selected_row)

    import cv2
    import numpy as np

    def detect_90_degree_corners_and_text(self, image_path):
        # Load the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the image for visualization
        boxed_image = image.copy()

        # Store details for boxes with 90-degree corners
        box_details = []

        def calculate_angle(pt1, pt2, pt3):
            """
            Calculate the angle (in degrees) between three points.
            pt1, pt2, pt3 are points in the format (x, y).
            """
            vec1 = np.array(pt1) - np.array(pt2)
            vec2 = np.array(pt3) - np.array(pt2)
            angle = np.degrees(np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))
            return angle

        for contour in contours:
            # Approximate the contour to simplify the shape
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the shape has 4 corners (indicating a rectangle or square)
            if len(approx) == 4:
                corners = [tuple(point[0]) for point in approx]

                # Calculate bounding box dimensions
                _, _, width, height = cv2.boundingRect(approx)

                # Filter out boxes with height < 20
                if height < 20:
                    continue

                # Calculate angles at each corner
                angles = []
                for i in range(4):
                    pt1 = corners[i - 1]
                    pt2 = corners[i]
                    pt3 = corners[(i + 1) % 4]
                    angle = calculate_angle(pt1, pt2, pt3)
                    angles.append(angle)

                # Check if all angles are approximately 90 degrees
                if all(85 <= angle <= 95 for angle in angles):
                    # Draw the rectangle on the image
                    cv2.drawContours(boxed_image, [approx], 0, (0, 255, 0), 2)

                    # Save box details
                    box_details.append({
                        "corners": corners,
                        "angles": angles,
                        "width": width,
                        "height": height
                    })

        # Return results
        return {
            "box_details": box_details,
        }


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProductPrototypeTool()
    window.show()
    sys.exit(app.exec_())
