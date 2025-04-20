# 🔬 Medical Image Classifier

![Medical Image Banner](https://img.shields.io/badge/Medical%20Image-Classifier-blue?style=for-the-badge&logo=microscope)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/spaces)

A deep learning-based medical image classification system that can identify 21 different blood cell types and medical conditions using a fine-tuned ResNet152 model. This project provides a user-friendly web interface built with Streamlit for medical image analysis.

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation Instructions](#-installation-instructions)
- [Running Locally](#-running-locally)
- [Deployment](#-deployment)
- [License](#-license)
- [Contributors](#-contributors)

## 🔍 Overview

This project implements a medical image classification system using deep learning techniques. The system is capable of classifying blood cell microscopy images into 21 different categories, each representing different blood cell types or medical conditions. The application provides a user-friendly web interface for uploading and analyzing medical images, displaying confidence scores and detailed descriptions of detected conditions.

The model was built using transfer learning with ResNet152 architecture pre-trained on ImageNet and fine-tuned on a specialized medical image dataset. The application provides detailed information about each identified condition, making it a valuable educational tool for medical professionals in training.

## ✨ Features

- 🏥 Classification of 21 different blood cell types and medical conditions
- 📊 Confidence scores for each prediction
- 📝 Detailed descriptions of each condition
- 🖼️ Sample images for testing and demonstration
- 📁 Upload custom images for analysis
- 📑 Comprehensive result visualization with bar charts
- 📤 Option to download analysis reports
- 📋 Easy-to-copy results for sharing
- 📱 Responsive design for various device sizes

## 🧠 Model Architecture

The classification model uses a **ResNet152** architecture with the following characteristics:

- **Base Model**: ResNet152 pre-trained on ImageNet
- **Transfer Learning**: Fine-tuned on medical images
- **Input Size**: 224×224 RGB images
- **Output**: 21 classes representing different blood cell types and conditions
- **Training**: Using data augmentation, class balancing, and learning rate decay
- **Performance**: ~95% training accuracy, ~92% validation accuracy, ~91% test accuracy
- **Cloud Deployment**: Model hosted on Hugging Face for on-demand loading

The detailed model development process can be found in the `Model.ipynb` notebook, which includes:

- Data preprocessing and augmentation steps
- Model architecture setup and fine-tuning
- Training process with performance metrics
- Model evaluation and visualization of results
- Saving the model for deployment

## 🗃️ Dataset

The model was trained on a specialized medical image dataset containing microscopy images of various blood cell types and medical conditions. The dataset was preprocessed, augmented, and balanced to ensure robust model performance across all 21 classes:

- ABE: Abetalipoproteinemia
- ART: Artifact
- BAS: Basophilia
- BLA: Blast Cells
- EBO: Elliptocytosis & Ovalocytosis
- EOS: Eosinophilia
- FGC: Familial Hypercholesterolemia
- HAC: Hairy Cell Leukemia
- KSC: Kawasaki Syndrome
- LYI: Lymphocytic Infiltration
- LYT: Lymphocytes
- MMZ: Marginal Zone Lymphoma
- MON: Monocytes
- MYB: Myeloblasts
- NGB: Neutrophilic Band Forms
- NGS: Neutrophilic Segmented
- NIF: Neutrophilic Infiltration
- OTH: Other blood cell abnormalities
- PEB: Peripheral Erythroblastosis
- PLM: Plasma Cells
- PMO: Promyelocytes

## 📁 Project Structure

```
Medical-Image-Classifier/
├── app.py                           # Main Streamlit application
├── requirements.txt                 # Project dependencies
├── Model.ipynb                      # Notebook for model development
├── best_resnet152_model.keras       # Trained model
├── resnet152_model_architecture.json # Model architecture
├── resnet152_model.weights.h5       # Model weights
├── class_indices.json               # Class mapping
├── abbreviations.csv                # Class abbreviations and full names
├── README.md                        # Project documentation
├── UserManual.md                    # Detailed user guide
├── LICENSE                          # License file
└── Sample Images/                   # Sample images for testing
    ├── ABE.jpeg
    ├── ART.jpeg
    ├── BAS.jpeg
    └── ...
```

## 🔧 Installation Instructions

### Prerequisites

- Python 3.8 or higher
- Git with LFS support (optional since models can be downloaded dynamically)
- 4GB+ RAM recommended
- Internet connection (for initial model download)

### Clone the Repository

You can clone our repository using Git:

1. Install Git LFS (optional but recommended):
   ```bash
   # For Windows
   git lfs install
   
   # For Linux/macOS
   brew install git-lfs  # macOS with Homebrew
   sudo apt-get install git-lfs  # Ubuntu/Debian
   git lfs install
   ```

2. Clone the repository:
   ```bash
   git clone https://huggingface.co/spaces/Jaya6061/Medical_Image_Classification_System
   cd Medical_Image_Classification_System
   ```

### Set up Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# For Windows
venv\Scripts\activate

# For Linux/macOS
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### About Model Files

Our application now supports two ways to access the model:

1. **On-Demand Download**: The app will automatically download the model from Hugging Face when first needed
2. **Pre-Downloaded**: If you have the model files locally, the app will use those instead

This approach eliminates the need to download large model files during installation.

## 🚀 Running Locally

Once you've completed the installation, you can run the application locally:

```bash
streamlit run app.py
```

The application will be available at http://localhost:8501 in your web browser.

## 🌐 Deployment

### Hugging Face Spaces

This project is deployed on Hugging Face Spaces with dynamic model loading:

1. Visit our [Hugging Face Space](https://huggingface.co/spaces/Jaya6061/Medical_Image_Classification_System)
2. The model will automatically download from our repository the first time it's needed
3. Upload your medical images and get instant analysis

### Deployment Innovations

- **Dynamic Model Loading**: The model is downloaded on-demand from [Hugging Face Model Hub](https://huggingface.co/Jaya6061/resnet152-medical-model)
- **Progress Tracking**: Users see download progress for transparency
- **Caching System**: Downloaded model is cached for future use
- **Graceful Degradation**: Fallback to sample data if connection issues occur

### Deployment Challenges and Solutions

- **File Size Limitations**: The model files exceed the size limits of Streamlit Cloud
- **Solution**: We use Hugging Face Spaces with dynamic model loading from the Model Hub
- **User Experience**: Progress indicators and clear status messages during model loading
- **Alternative**: For self-hosting, consider Docker containers or cloud VM instances

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributors

- **Jaya Vardhan Reddy** - Model development and application design
- **Preetham** - Data preprocessing and application deployment

### 💭 Thoughts

Working on this Medical Image Classifier project has been an incredible journey. The integration of deep learning with practical medical applications demonstrates how AI can support healthcare professionals. What impressed me most was the careful attention to user experience—making complex technology accessible through an intuitive interface.

The decision to use ResNet152 with transfer learning was excellent for this application, balancing accuracy with computational efficiency. The comprehensive sample image collection provides users with valuable context for understanding the classifications.

Deploying through Hugging Face Spaces was a smart solution to the file size challenges, ensuring the application remains accessible to anyone who needs it without sacrificing model quality. This project has tremendous potential for educational use in medical training environments, and I'm proud to have contributed to making this technology more accessible.
