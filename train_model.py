import cv2
import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report

class FireDetectorTrainer:
    def __init__(self, epochs=10):
        self.training_path = Path('data/training')
        self.test_path = Path('data/test')
        self.fire_path = self.training_path / 'fire'
        self.smoke_path = self.training_path / 'smoke'
        self.normal_path = self.training_path / 'normal'
        self.model_path = Path('models')
        self.model_path.mkdir(exist_ok=True)
        self.epochs = epochs
        self.class_map = {'fire': 0, 'smoke': 1, 'normal': 2}
        self.class_names = ['fire', 'smoke', 'normal']

    def extract_features(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    def get_image_files(self, folder):
        return list(folder.glob('*.jpg')) + list(folder.glob('*.png'))

    def load_dataset(self, base_path):
        X, y = [], []
        for cls in self.class_names:
            cls_path = base_path / cls
            for img_file in self.get_image_files(cls_path):
                img = cv2.imread(str(img_file))
                if img is not None:
                    X.append(self.extract_features(img))
                    y.append(self.class_map[cls])
        return np.array(X), np.array(y)

    def process_training_data(self):
        print("Loading training data (whole images)...")
        X_train, y_train = self.load_dataset(self.training_path)
        print(f"Loaded {len(y_train)} training samples.")
        X_test, y_test = self.load_dataset(self.test_path)
        print(f"Loaded {len(y_test)} test samples.")

        if len(y_train) == 0:
            print("No training data found. Please add images to data/training.")
            return
        if len(y_test) == 0:
            print("Warning: No test data found. Accuracy will not be reported.")

        clf = SVC(kernel='rbf', probability=True)
        best_acc = 0
        best_model = None
        for epoch in range(1, self.epochs + 1):
            X_train_shuf, y_train_shuf = shuffle(X_train, y_train, random_state=epoch)
            clf.fit(X_train_shuf, y_train_shuf)
            if len(y_test) > 0:
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(f"Epoch {epoch}/{self.epochs} - Test Accuracy: {acc*100:.2f}%")
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
    trainer = FireDetectorTrainer(epochs=20)  # You can set epochs here
    trainer.process_training_data()

if __name__ == "__main__":
    main() 