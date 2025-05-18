# Fire and Smoke Detection System

This project implements a real-time fire and smoke detection system using OpenCV, scikit-learn, and a machine learning classifier (SVM). The system can process video input from a file and detect fire and smoke in real-time using a trained model.

## Features

- Real-time fire and smoke detection using a trained SVM classifier
- Learns from your own images (not just color-based)
- Supports both JPG and PNG images
- Training and test set support for accuracy evaluation
- Multiple epochs for improved training
- Tkinter-based GUI for easy video selection and result display

## Directory Structure

```
data/
  training/
    fire/
    smoke/
    normal/
  test/
    fire/
    smoke/
    normal/
models/
```

- Place your training images in the appropriate `data/training/` subfolders.
- Place your test images in the appropriate `data/test/` subfolders.
- Both `.jpg` and `.png` images are supported.

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- scikit-learn
- Pillow

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip3 install -r requirements.txt
```

3. Make sure the required directories exist (these are created automatically, but you can check):
```bash
mkdir -p data/training/fire data/training/smoke data/training/normal data/test/fire data/test/smoke data/test/normal models
```

## Usage

### 1. Prepare Your Data
- Add your fire, smoke, and normal images to the respective folders under `data/training/` and `data/test/`.
- The more diverse and representative your images, the better the model will perform.

### 2. Train the Model
Run the training script to extract features, train the SVM classifier for multiple epochs, and evaluate accuracy:
```bash
python3 train_model.py
```
- The script will print accuracy after each epoch and save the best model to `models/svm_model.pkl`.
- It will also print a classification report at the end.

### 3. Run the Detection GUI
After training, launch the GUI to detect fire and smoke in videos:
```bash
python3 fire_detector_gui.py
```
- Use the "Select Video" button to choose a video file.
- Click "Play" to start detection. The GUI will display bounding boxes for detected fire and smoke regions in real time.
- Click "Pause" or "Stop" as needed.

## How It Works

- **Training:**
  - Extracts color histogram features from each image.
  - Trains an SVM classifier for multiple epochs, shuffling data each time.
  - Evaluates accuracy on the test set after each epoch.
  - Saves the best model.

- **Detection:**
  - The GUI loads the trained SVM model.
  - Each video frame is scanned using a sliding window.
  - Each region is classified as fire, smoke, or normal using the SVM.
  - Detected regions are highlighted with bounding boxes and labels.

## Notes

- The system does not rely on simple color thresholding; it learns from your images.
- For best results, provide a diverse and well-labeled dataset.
- You can increase the number of epochs in `train_model.py` for longer training.
- If you add more images, re-run the training script to update the model.

## Troubleshooting

- If you see an error about missing `svm_model.pkl`, make sure you have run the training script and that your data folders contain images.
- If you encounter any other errors, check that all dependencies are installed and that your Python version is compatible.

## License

MIT License 