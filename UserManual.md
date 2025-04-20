# 📘 Medical Image Classifier - User Manual

Welcome to the Medical Image Classifier user manual! This document provides detailed instructions on how to use the application effectively.

## 📋 Table of Contents
- [Getting Started](#-getting-started)
- [Application Interface](#-application-interface)
- [Image Classification](#-image-classification)
- [Understanding Results](#-understanding-results)
- [Sample Images](#-sample-images)
- [Troubleshooting](#-troubleshooting)
- [Technical Specifications](#-technical-specifications)
- [FAQs](#-faqs)

## 🚀 Getting Started

### Access Options

You can access the Medical Image Classifier in two ways:

1. **Online (Recommended)**: Visit our [Hugging Face Space](https://huggingface.co/spaces/Jaya6061/Medical_Image_Classification_System) to use the application without any installation.
   - On first use, the model will automatically download from our Hugging Face [repository](https://huggingface.co/Jaya6061/resnet152-medical-model)
   - You'll see a progress bar during the download process
   - All subsequent visits will use the cached model for faster load times

2. **Locally**: Follow these instructions to run on your machine:
   - Make sure you have installed all prerequisites (see README.md)
   - Activate your virtual environment
   - Run `streamlit run app.py`
   - Open your browser at http://localhost:8501
   - The app will automatically download the model from Hugging Face if not found locally

### System Requirements

For optimal performance when running locally:
- **Operating System**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+
- **RAM**: Minimum 4GB, 8GB recommended
- **CPU**: Multi-core processor
- **Disk Space**: At least 500MB free space
- **Browser**: Chrome, Firefox, or Edge (latest versions)

## 🖥️ Application Interface

The application interface is divided into several sections:

### Main Navigation
- **Image Classification tab**: Upload and analyze medical images
- **About the Model tab**: Learn about the model architecture and classes

### Sidebar
- **About**: Brief description of the application
- **Instructions**: How to use the application
- **Model Performance**: Statistics on model accuracy

## 🔍 Image Classification

The image classification feature allows you to analyze medical images in two ways:

### 1. Upload Your Image

1. Navigate to the "Image Classification" tab
2. Select the "Upload Your Image" sub-tab
3. Click "Browse files" or drag and drop your image (supported formats: JPG, JPEG, PNG)
4. Once uploaded, your image will be displayed
5. Click the "🔍 Classify Image" button
6. Wait for the progress bar to complete (typically a few seconds)
7. View your results below the image

### 2. Use Sample Images

1. Navigate to the "Image Classification" tab
2. Select the "Use Sample Image" sub-tab
3. Choose a sample image from the dropdown menu
4. The selected sample image will be displayed with its category description
5. Click the "🔍 Analyze Sample" button
6. Wait for the progress bar to complete
7. View your results below the image

## 📊 Understanding Results

After classification, the application displays results in a structured format:

### Primary Diagnosis
- The most likely condition with confidence percentage
- Detailed description of the condition

### Confidence Levels
- Bar chart showing confidence percentages for top 3 predictions
- Higher confidence levels are displayed with appropriate color coding

### Alternative Diagnoses
- Secondary conditions that might be present
- Descriptions and confidence levels for these conditions

### Report Options
- **Download Report**: Save results as a text file
- **Copy Results**: Copy formatted results to clipboard for sharing

## 🖼️ Sample Images

The application includes sample images for all 21 medical conditions:

- **ABE**: Abetalipoproteinemia samples
- **ART**: Artifact samples
- **BAS**: Basophilia samples
- **BLA**: Blast Cells samples
- **EBO**: Elliptocytosis & Ovalocytosis samples
- **EOS**: Eosinophilia samples
- **FGC**: Familial Hypercholesterolemia samples
- **HAC**: Hairy Cell Leukemia samples
- **KSC**: Kawasaki Syndrome samples
- **LYI**: Lymphocytic Infiltration samples
- **LYT**: Lymphocytes samples
- **MMZ**: Marginal Zone Lymphoma samples
- **MON**: Monocytes samples
- **MYB**: Myeloblasts samples
- **NGB**: Neutrophilic Band Forms samples
- **NGS**: Neutrophilic Segmented samples
- **NIF**: Neutrophilic Infiltration samples
- **OTH**: Other blood cell abnormalities samples
- **PEB**: Peripheral Erythroblastosis samples
- **PLM**: Plasma Cells samples
- **PMO**: Promyelocytes samples

Each sample image represents a typical example of the respective condition to help users understand what each category looks like.

## 🛠️ Troubleshooting

### Common Issues and Solutions

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Application won't start locally | Missing dependencies | Run `pip install -r requirements.txt` |
| Image won't upload | File too large or unsupported format | Resize image to under 5MB and ensure it's JPG, JPEG, or PNG |
| Classification takes too long | Slow internet or limited CPU resources | Try using sample images or wait longer for results |
| Model gives unexpected results | Low-quality or non-medical images | Ensure images are clear and are actual microscopy images |
| "Model loading error" message | Network issues | Check your internet connection; the app will attempt to download the model from Hugging Face |
| Previous results show with new images | UI state not cleared | The app now automatically clears previous results when a new image is uploaded |
| TensorFlow warning messages | Normal initialization warnings | These are suppressed in the latest version and won't affect functionality |

### Error Messages

- **"Could not load model weights"**: The application will use a mock model for demonstration
- **"Error loading sample image"**: Try selecting a different sample image
- **"Failed to process image"**: The image might be corrupted or in an unsupported format

## 🧪 Technical Specifications

### Model Details

- **Architecture**: ResNet152 (deep residual network with 152 layers)
- **Input**: 224x224 RGB images
- **Output**: 21 classes with confidence scores
- **Performance Metrics**:
  - Training Accuracy: ~95%
  - Validation Accuracy: ~92%
  - Test Accuracy: ~93%

### Data Processing

Images undergo the following preprocessing steps:
1. Resizing to 224x224 pixels
2. RGB channel normalization
3. Tensor conversion
4. ResNet-specific preprocessing

## ❓ FAQs

**Q: Is this application intended for medical diagnosis?**  
A: No, this application is for educational purposes only. Always consult with a qualified healthcare professional for medical advice and diagnosis.

**Q: How accurate is the model?**  
A: The model achieves approximately 91% accuracy on test data, but accuracy may vary depending on image quality and clarity.

**Q: Can I use my own images?**  
A: Yes, you can upload your own medical microscopy images for analysis.

**Q: How can I improve classification results?**  
A: Use clear, high-resolution microscopy images with good lighting and focus.

**Q: Is my data private when using the application?**  
A: When using the local version, your images never leave your computer. On Hugging Face Spaces, standard privacy policies apply.

**Q: Can I use this project for my own research?**  
A: Yes, this project is open-source under the MIT License. Please provide proper attribution.

## 📞 Support

For additional support, please:
- Open an issue on our GitHub repository
- Contact the contributors directly:
  - Jaya Vardhan Reddy
  - Preetham

## 💬 From the Development Team

> "This project represents our commitment to bridging the gap between advanced AI technologies and practical medical applications. We designed this tool to be not just accurate, but accessible and educational. Whether you're a medical student, researcher, or healthcare professional, we hope this application serves as both a practical tool and a learning resource in your journey."
>
> "The 21 blood cell categories represented in this application cover a wide range of medical conditions, each with unique characteristics and clinical significance. By providing detailed descriptions and visual examples, we aim to enhance understanding of these conditions beyond simple classification."
>
> "We welcome your feedback and suggestions for improvement as we continue to refine and expand this tool."
>
> — The Medical Image Classifier Team
