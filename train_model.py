import cv2
import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

class FireDetectorTrainer:
    def __init__(self, epochs=10, patch_size=64, step_size=32, batch_size=2048):
        self.dataset_path = Path('Dataset')
        self.splits = ['train', 'valid', 'test']
        self.patch_size = patch_size
        self.step_size = step_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = Path('models')
        self.model_path.mkdir(exist_ok=True)
        self.class_names = ['Fire', 'Smoke']
        self.class_map = {'Fire': 0, 'Smoke': 1}
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
                if int(cls_id) not in self.yolo_class_map:
                    continue
                x_c, y_c, w, h = x_c * img_w, y_c * img_h, w * img_w, h * img_h
                x1 = int(x_c - w / 2)
                y1 = int(y_c - h / 2)
                x2 = int(x_c + w / 2)
                y2 = int(y_c + h / 2)
                boxes.append({'class': int(cls_id), 'bbox': (x1, y1, x2, y2)})
        return boxes

    def extract_patches(self, img, boxes):
        h, w, _ = img.shape
        X, y = [], []
        # Only extract fire/smoke patches
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
        return X, y

    def load_dataset(self, split):
        X, y = [], []
        images_dir, image_files, label_files = self.get_image_and_label_files(split)
        for img_file in tqdm(image_files, desc=f'Loading {split} images'):
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
        print("Loading training data (YOLO fire/smoke patches only)...")
        X_train, y_train = self.load_dataset('train')
        print(f"Loaded {len(y_train)} training patches.")
        X_test, y_test = self.load_dataset('valid')
        print(f"Loaded {len(y_test)} validation patches.")

        if len(y_train) == 0:
            print("No training data found. Please check your dataset.")
            return
        if len(y_test) == 0:
            print("Warning: No validation data found. Accuracy will not be reported.")

        # Initialize classifier with static learning
        clf = SGDClassifier(loss='hinge', max_iter=1, tol=None, warm_start=True, n_jobs=-1)
        
        print(f"Training for {self.epochs} epoch(s)...")
        for epoch in tqdm(range(1, self.epochs + 1), desc='Epochs'):
            X_train_shuf, y_train_shuf = shuffle(X_train, y_train, random_state=epoch)
            # Train in batches
            for i in range(0, len(X_train_shuf), self.batch_size):
                X_batch = X_train_shuf[i:i+self.batch_size]
                y_batch = y_train_shuf[i:i+self.batch_size]
                if epoch == 1 and i == 0:
                    clf.partial_fit(X_batch, y_batch, classes=np.array([0, 1]))
                else:
                    clf.partial_fit(X_batch, y_batch)
            
            if len(y_test) > 0:
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                tqdm.write(f"Epoch {epoch}/{self.epochs} - Validation Accuracy: {acc*100:.2f}%")

        # Save the model
        with open(self.model_path / 'svm_model.pkl', 'wb') as f:
            pickle.dump(clf, f)
        print(f"Model saved to {self.model_path / 'svm_model.pkl'}")
        if len(y_test) > 0:
            print("Classification report:")
            print(classification_report(y_test, clf.predict(X_test), target_names=self.class_names))

def main():
    trainer = FireDetectorTrainer(epochs=10, patch_size=64, step_size=32, batch_size=2048)
    trainer.process_training_data()

if __name__ == "__main__":
    main() 