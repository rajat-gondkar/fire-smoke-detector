import cv2
import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report
import random

class FireDetectorTrainer:
    def __init__(self, epochs=1, patch_size=64, step_size=32, normal_patches_per_image=5):
        self.dataset_path = Path('Dataset')
        self.splits = ['train', 'valid', 'test']
        self.patch_size = patch_size
        self.step_size = step_size
        self.epochs = epochs
        self.normal_patches_per_image = normal_patches_per_image
        self.model_path = Path('models')
        self.model_path.mkdir(exist_ok=True)
        self.class_names = ['Fire', 'Smoke', 'Normal']
        self.class_map = {'Fire': 0, 'Smoke': 1, 'Normal': 2}
        self.yolo_class_map = {0: 0, 1: 1}  # YOLO class 0->Fire, 1->Smoke

    def extract_features(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    def get_image_and_label_files(self, split):
        images_dir = self.dataset_path / split / 'images'
        labels_dir = self.dataset_path / split / 'labels'
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        label_files = {f.stem: f for f in labels_dir.glob('*.txt')}
        return images_dir, image_files, label_files

    def parse_yolo_labels(self, label_file, img_w, img_h):
        boxes = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, x_c, y_c, w, h = map(float, parts)
                x_c, y_c, w, h = x_c * img_w, y_c * img_h, w * img_w, h * img_h
                x1 = int(x_c - w / 2)
                y1 = int(y_c - h / 2)
                x2 = int(x_c + w / 2)
                y2 = int(y_c + h / 2)
                boxes.append({'class': int(cls_id), 'bbox': (x1, y1, x2, y2)})
        return boxes

    def patch_overlaps(self, patch, boxes):
        px1, py1, px2, py2 = patch
        for box in boxes:
            bx1, by1, bx2, by2 = box['bbox']
            # Calculate intersection
            ix1 = max(px1, bx1)
            iy1 = max(py1, by1)
            ix2 = min(px2, bx2)
            iy2 = min(py2, by2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            intersection = iw * ih
            patch_area = (px2 - px1) * (py2 - py1)
            if patch_area == 0:
                continue
            if intersection / patch_area > 0.2:  # >20% overlap
                return True
        return False

    def extract_patches(self, img, boxes):
        h, w, _ = img.shape
        X, y = [], []
        # Extract fire/smoke patches
        for box in boxes:
            x1, y1, x2, y2 = box['bbox']
            # Center crop to patch size
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            px1 = max(0, cx - self.patch_size // 2)
            py1 = max(0, cy - self.patch_size // 2)
            px2 = min(w, px1 + self.patch_size)
            py2 = min(h, py1 + self.patch_size)
            patch = img[py1:py2, px1:px2]
            if patch.shape[0] == self.patch_size and patch.shape[1] == self.patch_size:
                X.append(self.extract_features(patch))
                y.append(self.yolo_class_map[box['class']])
        # Extract normal patches
        normal_count = 0
        attempts = 0
        max_attempts = self.normal_patches_per_image * 10
        while normal_count < self.normal_patches_per_image and attempts < max_attempts:
            px1 = random.randint(0, w - self.patch_size)
            py1 = random.randint(0, h - self.patch_size)
            px2 = px1 + self.patch_size
            py2 = py1 + self.patch_size
            patch_box = (px1, py1, px2, py2)
            if not self.patch_overlaps(patch_box, boxes):
                patch = img[py1:py2, px1:px2]
                if patch.shape[0] == self.patch_size and patch.shape[1] == self.patch_size:
                    X.append(self.extract_features(patch))
                    y.append(2)  # Normal
                    normal_count += 1
            attempts += 1
        return X, y

    def load_dataset(self, split):
        X, y = [], []
        images_dir, image_files, label_files = self.get_image_and_label_files(split)
        for img_file in image_files:
            label_file = label_files.get(img_file.stem)
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            h, w, _ = img.shape
            boxes = []
            if label_file and label_file.exists():
                boxes = self.parse_yolo_labels(label_file, w, h)
            Xi, yi = self.extract_patches(img, boxes)
            X.extend(Xi)
            y.extend(yi)
        return np.array(X), np.array(y)

    def process_training_data(self):
        print("Loading training data (YOLO patches)...")
        X_train, y_train = self.load_dataset('train')
        print(f"Loaded {len(y_train)} training patches.")
        X_test, y_test = self.load_dataset('valid')
        print(f"Loaded {len(y_test)} validation patches.")

        if len(y_train) == 0:
            print("No training data found. Please check your dataset.")
            return
        if len(y_test) == 0:
            print("Warning: No validation data found. Accuracy will not be reported.")

        clf = SVC(kernel='rbf', probability=True)
        best_acc = 0
        best_model = None
        for epoch in range(1, self.epochs + 1):
            X_train_shuf, y_train_shuf = shuffle(X_train, y_train, random_state=epoch)
            clf.fit(X_train_shuf, y_train_shuf)
            if len(y_test) > 0:
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(f"Epoch {epoch}/{self.epochs} - Validation Accuracy: {acc*100:.2f}%")
                if acc > best_acc:
                    best_acc = acc
                    best_model = pickle.dumps(clf)
            else:
                print(f"Epoch {epoch}/{self.epochs} - Training complete.")
        if best_model is not None:
            clf = pickle.loads(best_model)
        # Save the best model
        with open(self.model_path / 'svm_model.pkl', 'wb') as f:
            pickle.dump(clf, f)
        print(f"Best model saved to {self.model_path / 'svm_model.pkl'}")
        if len(y_test) > 0:
            print("Classification report:")
            print(classification_report(y_test, clf.predict(X_test), target_names=self.class_names))


def main():
    trainer = FireDetectorTrainer(epochs=1, patch_size=64, step_size=32, normal_patches_per_image=5)
    trainer.process_training_data()

if __name__ == "__main__":
    main() 