# Fire and Smoke Detection System

This project implements a real-time fire and smoke detection system using OpenCV, scikit-learn, and a machine learning classifier (SVM). The system is now designed to work with a YOLO v8-style dataset, using bounding box annotations to extract patches for fire, smoke, and normal (background) classes.

## Features

- Real-time fire and smoke detection using a trained SVM classifier
- Learns from your own images and bounding box annotations (not just color-based)
- Supports YOLO v8 dataset format for object detection
- Extracts patches for fire, smoke, and normal (background) classes
- Training and validation set support for accuracy evaluation
- Multiple epochs for improved training (default: 20)
- Progress bar for both data loading and training epochs
- Tkinter-based GUI for easy video selection and result display

## Directory Structure

```
Dataset/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
  data.yaml
models/
```

- Place your YOLO v8 dataset in the `Dataset/` directory as shown above.
- Each image in `images/` should have a corresponding `.txt` file in `labels/` with YOLO v8 bounding box annotations.
- The `data.yaml` file should define your classes (e.g., Fire, Smoke).

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- scikit-learn
- Pillow
- tqdm

## Installation & Setup

1. **(Optional but recommended) Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset:**
   - Download or organize your YOLO v8 dataset as shown above.
   - Make sure each image has a corresponding label file.
   - The `data.yaml` file should list your classes (e.g., Fire, Smoke).

## Usage

### 1. Train the Model
Run the training script to extract patches, train the SVM classifier for 20 epochs, and evaluate accuracy:
```bash
python train_model.py
```
- The script will show a progress bar for image loading and for each epoch.
- Validation accuracy is printed after each epoch.
- The best model is saved to `models/svm_model.pkl`.
- A classification report is printed at the end.

### 2. Run the Detection GUI
After training, launch the GUI to detect fire and smoke in videos:
```bash
python fire_detector_gui.py
```
- Use the "Select Video" button to choose a video file.
- Click "Play" to start detection. The GUI will display the predicted class for each frame.
- Click "Pause" or "Stop" as needed.

## How It Works

- **Training:**
  - For each image, extracts patches from bounding boxes (fire/smoke) and random non-overlapping patches (normal).
  - Extracts color histogram features from each patch.
  - Trains an SVM classifier for 20 epochs, shuffling data each time.
  - Evaluates accuracy on the validation set after each epoch.
  - Saves the best model.

- **Detection:**
  - The GUI loads the trained SVM model.
  - Each video frame is classified as fire, smoke, or normal using the SVM (on the whole frame or patch, depending on GUI version).
  - The predicted class is displayed on the frame.

## Notes

- The system does not rely on simple color thresholding; it learns from your annotated dataset.
- For best results, provide a diverse and well-labeled dataset.
- You can increase or decrease the number of epochs in `train_model.py`.
- If you add more images, re-run the training script to update the model.

## Troubleshooting

- If you see an error about missing `svm_model.pkl`, make sure you have run the training script and that your dataset is correctly formatted.
- If you encounter any other errors, check that all dependencies are installed and that your Python version is compatible.
- If you get a `ModuleNotFoundError`, make sure you are in your virtual environment and have installed all requirements.

## License

MIT License 