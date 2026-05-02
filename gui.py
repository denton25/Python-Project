import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading
import cv2
import time
import torch

from main import load_model, predict_word_fn, segment_image
from train_pytorch import train_model


class OCR_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Camren's Handy OCR Text Extractor!")
        self.root.geometry("1400x900")
        self.root.configure(bg="black")

        self.model = None
        self.device = None
        self.current_image_path = None
        self.training_active = False

        # ============================================================
        # LEFT SIDE — BIGGER IMAGE PREVIEW
        # ============================================================
        left_frame = tk.Frame(self.root, bg="black")
        left_frame.pack(side="left", padx=20, pady=20, fill="both", expand=True)

        self.preview_frame = tk.Frame(left_frame, bg="gray", width=700, height=700)
        self.preview_frame.pack(fill="both", expand=True)
        self.preview_frame.pack_propagate(False)

        self.image_label = tk.Label(self.preview_frame, bg="gray")
        self.image_label.pack(fill="both", expand=True)

        # ============================================================
        # RIGHT SIDE — OUTPUT + TRAINING LOG
        # ============================================================
        right_frame = tk.Frame(self.root, bg="black")
        right_frame.pack(side="right", padx=20, pady=20, fill="y")

        console_label = tk.Label(
            right_frame,
            text="Output / Training Log",
            font=("Arial", 16, "bold"),
            bg="black",
            fg="white"
        )
        console_label.pack()

        self.console = tk.Text(
            right_frame,
            width=55,
            height=35,
            font=("Consolas", 12),
            bg="lightyellow",
            state="disabled"
        )
        self.console.pack(fill="y")

        self.progress = ttk.Progressbar(
            right_frame,
            orient="horizontal",
            length=350,
            mode="determinate"
        )
        self.progress.pack(pady=10)

        self.spinner_label = tk.Label(
            right_frame,
            text="",
            font=("Arial", 16),
            bg="black",
            fg="white"
        )
        self.spinner_label.pack(pady=5)

        # ============================================================
        # BOTTOM BUTTON BAR
        # ============================================================
        btn_frame = tk.Frame(self.root, bg="black", height=70)
        btn_frame.pack(side="bottom", fill="x")
        btn_frame.pack_propagate(False)

        ttk.Button(btn_frame, text="Load Model", command=self.load_model_button).grid(row=0, column=0, padx=15)
        ttk.Button(btn_frame, text="Load Image", command=self.load_image).grid(row=0, column=1, padx=15)
        ttk.Button(btn_frame, text="Extract Text", command=self.extract_text).grid(row=0, column=2, padx=15)
        ttk.Button(btn_frame, text="Train Model", command=self.start_training_thread).grid(row=0, column=3, padx=15)
        ttk.Button(btn_frame, text="Debug Segmentation", command=self.debug_segmentation).grid(row=0, column=4, padx=15)

    # ============================================================
    # LOGGING
    # ============================================================
    def log(self, msg):
        self.console.config(state="normal")
        self.console.insert(tk.END, msg + "\n")
        self.console.config(state="disabled")
        self.console.see(tk.END)

    # ============================================================
    # TRAINING SYSTEM
    # ============================================================
    def spinner_animation(self):
        frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        i = 0
        while self.training_active:
            self.spinner_label.config(text=frames[i % len(frames)])
            i += 1
            time.sleep(0.1)

    def start_training_thread(self):
        if self.training_active:
            return

        self.training_active = True
        self.log("Starting training...")

        threading.Thread(target=self.spinner_animation, daemon=True).start()
        threading.Thread(target=self.run_training, daemon=True).start()

    def run_training(self):
        def log_callback(epoch, step, total_steps, loss):
            self.log(f"Epoch {epoch} Step {step}/{total_steps} Loss: {loss:.4f}")
            self.progress["value"] = (step / total_steps) * 100

        train_model(
            device="cuda" if torch.cuda.is_available() else "cpu",
            log_callback=log_callback
        )

        self.training_active = False
        self.spinner_label.config(text="")
        self.progress["value"] = 100
        self.log("Training complete! Model saved as model.pth")

    # ============================================================
    # MODEL LOADING
    # ============================================================
    def load_model_button(self):
        try:
            self.model, self.device = load_model()
            self.log("Model loaded successfully.")
        except Exception as e:
            self.log(f"Error loading model: {e}")

    # ============================================================
    # IMAGE LOADING
    # ============================================================
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not path:
            return

        self.current_image_path = path

        img = Image.open(path)
        self.preview_frame.update_idletasks()
        w = self.preview_frame.winfo_width()
        h = self.preview_frame.winfo_height()
        img.thumbnail((w, h))

        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

        self.log(f"Loaded image: {path}")

    # ============================================================
    # TEXT EXTRACTION
    # ============================================================
    def extract_text(self):
        if self.model is None:
            self.log("Error: Load a model first.")
            return

        if not self.current_image_path:
            self.log("Error: Load an image first.")
            return

        text, boxes = predict_word_fn(self.model, self.device, self.current_image_path)

        img = cv2.imread(self.current_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for (x, y, w, h) in boxes:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        img = Image.fromarray(img)
        self.preview_frame.update_idletasks()
        w = self.preview_frame.winfo_width()
        h = self.preview_frame.winfo_height()
        img.thumbnail((w, h))

        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

        self.log(f"Predicted: {text}")

    # ============================================================
    # SEGMENTATION DEBUG WINDOW
    # ============================================================
    def debug_segmentation(self):
        if not self.current_image_path:
            self.log("Load an image first.")
            return

        chars, _ = segment_image(self.current_image_path)
        if not chars:
            self.log("No characters found in segmentation.")
            return

        win = tk.Toplevel(self.root)
        win.title("Segmentation Debug")

        row = 0
        col = 0

        for ch in chars:
            img = Image.fromarray(ch)
            img = img.resize((80, 80))
            tk_img = ImageTk.PhotoImage(img)

            lbl = tk.Label(win, image=tk_img)
            lbl.image = tk_img
            lbl.grid(row=row, column=col, padx=5, pady=5)

            col += 1
            if col >= 10:
                col = 0
                row += 1


def start_gui():
    root = tk.Tk()
    app = OCR_GUI(root)
    root.mainloop()


if __name__ == "__main__":
    start_gui()


