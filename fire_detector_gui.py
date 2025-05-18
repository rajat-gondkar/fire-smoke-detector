import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import pickle
from pathlib import Path
import threading
import queue

class FireDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fire and Smoke Detection System")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.class_names = ['fire', 'smoke', 'normal']
        
        # Load trained SVM model
        self.model = self.load_model()
        if self.model is None:
            messagebox.showerror("Model Error", "Trained model not found. Please run train_model.py first.")
            self.root.destroy()
            return
        
        # Create GUI elements
        self.create_widgets()
        
    def load_model(self):
        try:
            with open(Path('models/svm_model.pkl'), 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=2, pady=10)
        ttk.Button(control_frame, text="Select Video", command=self.select_video).pack(side=tk.LEFT, padx=5)
        self.play_button = ttk.Button(control_frame, text="Play", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop", command=self.stop_video).pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(main_frame, text="Status: Ready")
        self.status_label.grid(row=2, column=0, columnspan=2, pady=5)
    
    def select_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if self.video_path:
            self.status_label.config(text=f"Status: Video selected - {self.video_path}")
            self.play_button.config(state=tk.NORMAL)
    
    def toggle_play(self):
        if not self.is_playing:
            self.start_video()
        else:
            self.pause_video()
    
    def start_video(self):
        if not self.video_path:
            return
        self.cap = cv2.VideoCapture(self.video_path)
        self.is_playing = True
        self.play_button.config(text="Pause")
        self.status_label.config(text="Status: Playing")
        self.process_thread = threading.Thread(target=self.process_video)
        self.process_thread.daemon = True
        self.process_thread.start()
        self.update_frame()
    
    def pause_video(self):
        self.is_playing = False
        self.play_button.config(text="Play")
        self.status_label.config(text="Status: Paused")
    
    def stop_video(self):
        self.is_playing = False
        if self.cap:
            self.cap.release()
        self.play_button.config(text="Play")
        self.status_label.config(text="Status: Stopped")
        self.video_label.config(image='')
    
    def process_video(self):
        while self.is_playing:
            ret, frame = self.cap.read()
            if not ret:
                self.is_playing = False
                break
            processed_frame = self.detect_objects(frame)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(processed_frame)
            img = ImageTk.PhotoImage(image=img)
            try:
                self.frame_queue.put(img, block=False)
            except queue.Full:
                pass
    
    def update_frame(self):
        if self.is_playing:
            try:
                img = self.frame_queue.get(block=False)
                self.video_label.config(image=img)
                self.video_label.image = img
            except queue.Empty:
                pass
            self.root.after(30, self.update_frame)
    
    def extract_features(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten().reshape(1, -1)
    
    def detect_objects(self, frame):
        # Use sliding window or region proposals for detection
        h, w, _ = frame.shape
        window_size = 64
        step_size = 32
        for y in range(0, h - window_size, step_size):
            for x in range(0, w - window_size, step_size):
                roi = frame[y:y+window_size, x:x+window_size]
                if roi.shape[0] != window_size or roi.shape[1] != window_size:
                    continue
                features = self.extract_features(roi)
                pred = self.model.predict(features)[0]
                if pred != 2:  # Not 'normal'
                    color = (0, 0, 255) if pred == 0 else (128, 128, 128)
                    label = self.class_names[pred].capitalize()
                    cv2.rectangle(frame, (x, y), (x+window_size, y+window_size), color, 2)
                    cv2.putText(frame, label, (x, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

def main():
    root = tk.Tk()
    app = FireDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 