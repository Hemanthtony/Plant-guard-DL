# PlantGuard: AI-Powered Plant Disease Detection System

## Overview

PlantGuard is an intelligent web application that uses deep learning and computer vision to identify plant species and detect diseases from uploaded images. The system combines the Plant.id API for plant identification with a custom-trained Convolutional Neural Network (CNN) for disease detection, providing farmers, gardeners, and plant enthusiasts with accurate and actionable insights about plant health.

## Features

- **Plant Identification**: Uses Plant.id API to identify plant species with high accuracy
- **Disease Detection**: Custom CNN model trained on extensive plant disease dataset
- **Health Assessment**: Provides health status and probability scores
- **Web Interface**: User-friendly web application for easy image upload and results
- **Fallback System**: Local model serves as backup when API is unavailable
- **Real-time Results**: Instant analysis and display of results

## Technology Stack

### Backend
- **Flask**: Python web framework for API endpoints
- **TensorFlow/Keras**: Deep learning framework for CNN model
- **Plant.id API**: External API for plant identification and health assessment
- **PIL (Pillow)**: Image processing library
- **NumPy**: Numerical computing library

### Frontend
- **HTML5**: Structure and content
- **CSS3**: Styling and responsive design
- **JavaScript**: Client-side functionality and API communication

### Machine Learning
- **Convolutional Neural Network (CNN)**: Custom architecture for disease classification
- **Data Augmentation**: Image preprocessing and augmentation techniques
- **Transfer Learning**: Optimized training approach

## Dataset

The model is trained on the "Plant Disease Detection" dataset from Kaggle, containing:
- **38 different plant disease classes**
- **Multiple plant species**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
- **Total images**: Thousands of labeled images for training and validation
- **Class distribution**: Balanced dataset with healthy and diseased samples

## Model Architecture

The CNN model consists of:
- **4 Convolutional layers** with increasing filter sizes (32, 64, 128, 128)
- **MaxPooling layers** for dimensionality reduction
- **Dropout regularization** (0.5) to prevent overfitting
- **Dense layers**: 512 neurons with ReLU activation
- **Output layer**: Softmax activation for multi-class classification

### Training Configuration
- **Input size**: 150x150 pixels
- **Batch size**: 32
- **Epochs**: 20
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss function**: Categorical cross-entropy
- **Data augmentation**: Rotation, shifting, shearing, zooming, flipping

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hemanthtony/Plant-guard-DL.git
   cd Plant-guard-DL
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset** (optional, for training)
   ```bash
   python download_dataset.py
   ```

5. **Train the model** (optional, pre-trained model included)
   ```bash
   python train_model.py
   ```

6. **Evaluate the model** (optional)
   ```bash
   python evaluate_model.py
   ```

## Usage

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:5000`

3. **Upload an image** of a plant using the web interface

4. **View results** including:
   - Plant species identification
   - Common names
   - Health status (healthy/diseased)
   - Confidence probabilities

### API Usage

The application provides a REST API endpoint:

**POST** `/predict`
- **Input**: Multipart form data with 'file' field containing image
- **Output**: JSON response with plant identification and health assessment

Example using curl:
```bash
curl -X POST -F "file=@plant_image.jpg" http://localhost:5000/predict
```

## Project Structure

```
Plant-guard-DL/
│
├── app.py                      # Flask application with API endpoints
├── train_model.py              # CNN model training script
├── evaluate_model.py           # Model evaluation script
├── count_images.py             # Dataset analysis script
├── download_dataset.py         # Dataset download script
│
├── index.html                  # Main web interface
├── styles.css                  # CSS styling
├── script.js                   # Frontend JavaScript
│
├── plant_disease_model.h5      # Trained CNN model
├── training_history.png        # Training visualization
├── requirements.txt            # Python dependencies
├── TODO.md                     # Project tasks and progress
└── README.md                   # This file
```

## Model Performance

- **Validation Accuracy**: ~85-90% (varies with dataset and training)
- **Supported Diseases**: 38 different plant disease classes
- **Image Requirements**: RGB images, minimum 150x150 pixels
- **Processing Time**: < 2 seconds per image on CPU

## API Integration

### Plant.id API
- **Purpose**: Plant identification and health assessment
- **Features**: Species recognition, common names, taxonomy
- **Fallback**: Local CNN model when API unavailable

### Local CNN Model
- **Purpose**: Disease classification when API fails
- **Classes**: 38 plant disease categories
- **Accuracy**: High confidence for trained diseases

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Enhancements

- [ ] Mobile application development
- [ ] Additional plant species support
- [ ] Real-time video analysis
- [ ] Treatment recommendations
- [ ] Multi-language support
- [ ] Cloud deployment
- [ ] Batch processing capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Plant.id API** for plant identification services
- **Kaggle** for the plant disease detection dataset
- **TensorFlow/Keras** for deep learning framework
- **Flask** for web framework

## Contact

**Hemanth Tony**
- GitHub: [@Hemanthtony](https://github.com/Hemanthtony)
- Project Repository: [Plant-guard-DL](https://github.com/Hemanthtony/Plant-guard-DL)

---

**Note**: This project is for educational and research purposes. Always consult with agricultural experts for professional plant disease diagnosis and treatment recommendations.
