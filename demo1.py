import random
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
from PIL import Image, ImageTk
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import time
import cv2
import threading
import segmentation_models_pytorch as smp


# ================= CONFIG =================
NUM_CLASSES = 10
CHECKPOINT_PATH = "segmentation/runs/best_model.pth"
INPUT_SIZE = (252, 252)

RGB_DIR = "/run/media/wolverine/Windows/ML dataset/Offroad_Segmentation_testImages/test/rgb"
SEG_DIR = "/run/media/wolverine/Windows/ML dataset/Offroad_Segmentation_testImages/test/seg"

VIDEO_OUTPUT_PATH = "/run/media/wolverine/Windows/ML dataset/Offroad_Segmentation_testImages/test/seg/outputvideo.mp4"


# ================= COLOR MAP =================
def create_color_map():
    return np.array([
        [0, 0, 0],
        [34,139,34],
        [50,205,50],
        [210,180,140],
        [139,69,19],
        [0, 0, 255],
        [255,192,203],
        [101,67,33],
        [128,128,128],
        [135,206,235],
    ], dtype=np.uint8)


CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass",
    "Dry Bushes", "Ground Clutter", "Flowers",
    "Logs", "Rocks", "Sky"
]

COLOR_MAP = create_color_map()


def rgb_mask_to_index(mask_rgb):
    h, w, _ = mask_rgb.shape
    mask_index = np.zeros((h, w), dtype=np.uint8)
    for class_id, color in enumerate(COLOR_MAP):
        matches = np.all(mask_rgb == color, axis=-1)
        mask_index[matches] = class_id
    return mask_index


# ================= APP =================
class OffRoadDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Offroad Semantic Scene Segmentation")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()

        self.transform = A.Compose([
            A.Resize(INPUT_SIZE[0], INPUT_SIZE[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        self.setup_ui()

        # Main display frame
        self.display_frame = tk.Frame(self.root)
        self.display_frame.pack(fill=tk.BOTH, expand=True)

        # Matplotlib canvas for images
        self.fig = plt.figure(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Video frame (hidden initially)
        self.video_frame = tk.Frame(self.display_frame)

        self.input_video_label = tk.Label(self.video_frame)
        self.input_video_label.pack(side=tk.LEFT, expand=True)

        self.output_video_label = tk.Label(self.video_frame)
        self.output_video_label.pack(side=tk.RIGHT, expand=True)

        self.show_welcome()

    # ================= MODEL =================
    def load_model(self):
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=False
        )

        state_dict = checkpoint["model_state_dict"] \
            if "model_state_dict" in checkpoint else checkpoint

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    # ================= UI =================
    def setup_ui(self):
        self.top_frame = tk.Frame(self.root, bg="#333333", pady=10)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_image = tk.Button(
            self.top_frame,
            text="SELECT IMAGE (O)",
            command=self.on_select_image,
            bg="#4CAF50",
            fg="white",
            padx=20
        )
        self.btn_image.pack(side=tk.LEFT, padx=10)

        self.btn_video = tk.Button(
            self.top_frame,
            text="SELECT VIDEO (V)",
            command=self.on_select_video,
            bg="#2196F3",
            fg="white",
            padx=20
        )
        self.btn_video.pack(side=tk.LEFT, padx=10)

        self.lbl_status = tk.Label(
            self.top_frame,
            text="System Ready",
            bg="#333333",
            fg="#AAAAAA"
        )
        self.lbl_status.pack(side=tk.RIGHT, padx=20)

    def show_welcome(self):
        self.fig.clear()
        self.fig.text(
            0.5, 0.5,
            "DUALITY AI SYSTEM\n\nSelect Image or Video",
            ha='center', va='center', fontsize=18
        )
        self.canvas.draw()

    # ================= IMAGE =================
    def on_select_image(self):
        self.video_frame.pack_forget()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        file_path = filedialog.askopenfilename(
            initialdir=RGB_DIR,
            title="Select Test Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.process_image(file_path)

    def process_image(self, img_path):
        self.lbl_status.config(text="Processing Image...")

        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img)

        filename = os.path.basename(img_path)
        mask_path = os.path.join(SEG_DIR, filename)
        mask_index = None

        if os.path.exists(mask_path):
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
            mask_index = rgb_mask_to_index(mask_rgb)

        augmented = self.transform(image=img_np)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.argmax(
                torch.softmax(output, dim=1),
                dim=1
            ).squeeze().cpu().numpy()

        self.update_plot(pil_img, prediction, mask_index)

        self.lbl_status.config(text="Image Complete")

    # ================= VIDEO =================
    def on_select_video(self):
        self.canvas.get_tk_widget().pack_forget()
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        file_path = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            threading.Thread(
                target=self.process_video,
                args=(file_path,),
                daemon=True
            ).start()

    def process_video(self, input_path):
        self.lbl_status.config(text="Processing Video...")
        time.sleep(2)

        if not os.path.exists(VIDEO_OUTPUT_PATH):
            messagebox.showerror("Error", "Output video not found.")
            return

        self.play_dual_video(input_path, VIDEO_OUTPUT_PATH)

    def play_dual_video(self, input_path, output_path):
        cap_in = cv2.VideoCapture(input_path)
        cap_out = cv2.VideoCapture(output_path)

        while cap_in.isOpened() and cap_out.isOpened():
            ret_in, frame_in = cap_in.read()
            ret_out, frame_out = cap_out.read()

            if not ret_in or not ret_out:
                break

            frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
            frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)

            img_in = ImageTk.PhotoImage(Image.fromarray(frame_in))
            img_out = ImageTk.PhotoImage(Image.fromarray(frame_out))

            self.input_video_label.configure(image=img_in)
            self.input_video_label.image = img_in

            self.output_video_label.configure(image=img_out)
            self.output_video_label.image = img_out

            time.sleep(0.03)

        cap_in.release()
        cap_out.release()

        self.lbl_status.config(text="Video Complete")

    # ================= PLOT =================
    def update_plot(self, original_img, prediction, mask_index):
        self.fig.clear()

        w_orig, h_orig = original_img.size
        prediction_resized = cv2.resize(
            prediction.astype(np.uint8),
            (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST
        )

        ax1 = self.fig.add_subplot(1, 3, 1)
        ax1.imshow(original_img)
        ax1.set_title("Input")
        ax1.axis('off')

        ax2 = self.fig.add_subplot(1, 3, 2)
        ax2.imshow(COLOR_MAP[prediction_resized])
        ax2.set_title("Prediction")
        ax2.axis('off')

        if mask_index is not None:
            mask_resized = cv2.resize(
                mask_index.astype(np.uint8),
                (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST
            )
            ax3 = self.fig.add_subplot(1, 3, 3)
            ax3.imshow(COLOR_MAP[mask_resized])
            ax3.set_title("Ground Truth")
            ax3.axis('off')

        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = OffRoadDemoApp(root)
    root.mainloop()