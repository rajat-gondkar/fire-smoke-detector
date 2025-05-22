# Fire and Smoke Detection System

A computer vision-based system for detecting fire and smoke in videos using machine learning. The system uses a static learning approach with color histogram features and SVM classification.

## Technical Overview

### Model Architecture
- **Classifier**: SGDClassifier (Stochastic Gradient Descent) with hinge loss (Linear SVM)
- **Feature Extraction**: HSV Color Histograms
  - 32x32 bins for Hue and Saturation channels
  - Normalized histogram values
- **Training Approach**: Static learning with fixed parameters
  - 10 epochs
  - Batch size: 2048
  - Patch size: 64x64 pixels
  - Step size: 32 pixels

### Technical Stack
- **Computer Vision**: OpenCV (cv2)
- **Machine Learning**: scikit-learn
- **Web Interface**: Streamlit
- **Data Processing**: NumPy
- **Progress Tracking**: tqdm

## Dataset Structure
```
Dataset/
├── train/
│   ├── images/
│   │   └── *.jpg
│   └── labels/
│       └── *.txt
├── valid/
│   ├── images/
│   │   └── *.jpg
│   └── labels/
│       └── *.txt
└── test/
    ├── images/
    │   └── *.jpg
    └── labels/
        └── *.txt
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd FireDetector
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training the Model
```bash
python train_model.py
```
The training process:
1. Loads images and YOLO format labels from the Dataset directory
2. Extracts patches from labeled regions (fire/smoke)
3. Computes HSV color histograms for each patch
4. Trains the SVM classifier for 10 epochs
5. Saves the trained model to `models/svm_model.pkl`

### 2. Running the Web Interface
```bash
streamlit run streamlit_app.py
```
The web interface:
1. Loads the trained model
2. Provides a video upload interface
3. Processes the video frame by frame
4. Displays predictions with confidence scores
5. Shows real-time detection results

## How It Works

### Training Process
1. **Data Loading**:
   - Images are loaded from the Dataset directory
   - YOLO format labels are parsed to get bounding boxes
   - Only patches containing fire/smoke are extracted

2. **Feature Extraction**:
   - Images are converted to HSV color space
   - 2D histograms are computed for Hue and Saturation channels
   - Histograms are normalized to [0,1] range
   - Features are flattened into 1D vectors

3. **Model Training**:
   - SGDClassifier with hinge loss is initialized
   - Training data is shuffled for each epoch
   - Model is trained in batches of 2048 samples
   - Validation accuracy is computed after each epoch

### Detection Process
1. **Frame Processing**:
   - Each video frame is processed independently
   - HSV color histograms are computed
   - Features are fed to the trained model

2. **Prediction**:
   - Model predicts class (Fire/Smoke)
   - Decision function provides confidence score
   - Results are displayed on the frame

## Performance Considerations
- Processing speed depends on:
  - Video resolution
  - Hardware capabilities
  - System resources
- The model uses static learning, so performance is consistent across runs
- No GPU acceleration is required

## Limitations
- Static learning approach may not capture complex patterns
- Relies heavily on color features
- No temporal information consideration
- Fixed number of epochs (10)

## Future Improvements
- Implement deep learning-based approach
- Add temporal information processing
- Include more feature types
- Add real-time video stream support
- Implement GPU acceleration

## Troubleshooting

### Common Issues
1. **Model not found**:
   - Ensure you've run `train_model.py` first
   - Check if `models/svm_model.pkl` exists

2. **Video processing errors**:
   - Verify video format (supported: mp4, avi, mov)
   - Check video file integrity

3. **Memory issues**:
   - Reduce batch size in `train_model.py`
   - Process lower resolution videos