import os
import logging
import warnings

# More aggressive TensorFlow warning suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 1 = no INFO, 2 = no WARNING, 3 = no ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom operations
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent TensorFlow from allocating all GPU memory at once
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use CPU only, remove this line if you want to use GPU

# Configure Python warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure logging more aggressively
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)  # absl is used by TensorFlow internally

import streamlit as st
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time
import io
import requests
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json

# Disable TensorFlow deprecation warnings
tf.get_logger().setLevel(logging.ERROR)

# Set page configuration
st.set_page_config(
    page_title="Medical Image Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown(""" 
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    .highlight {
        color: #1E88E5;
        font-weight: bold;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f5f5f5;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .class-label {
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to load model
@st.cache_resource
def load_model():
    try:
        # Local paths
        model_path = "best_resnet152_model.keras"
        class_indices_path = "class_indices.json"
        
        # Hugging Face URLs
        model_url = "https://huggingface.co/Jaya6061/resnet152-medical-model/resolve/main/best_resnet152_model.keras"
        class_indices_url = "https://huggingface.co/Jaya6061/resnet152-medical-model/raw/main/class_indices.json"
        
        # Download model from Hugging Face if it doesn't exist locally
        if not os.path.exists(model_path):
            # Create placeholders that we can clear later
            info_placeholder = st.empty()
            progress_placeholder = st.empty()
            success_placeholder = st.empty()
            
            with st.spinner("Downloading model from Hugging Face (this may take a few moments)..."):
                info_placeholder.info("Model not found locally. Downloading from Hugging Face...")
                try:
                    response = requests.get(model_url, stream=True)
                    response.raise_for_status()  # Check for any errors
                    
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 1024 * 1024  # 1 MB
                    
                    # Create a progress bar for downloading
                    progress_bar = progress_placeholder.progress(0)
                    
                    with open(model_path, "wb") as f:
                        dl = 0
                        for data in response.iter_content(block_size):
                            dl += len(data)
                            f.write(data)
                            if total_size > 0:
                                progress = int(100 * dl / total_size)
                                progress_bar.progress(progress)
                    
                    # Show success message briefly then clear it
                    success_placeholder.success("Model downloaded successfully!")
                    time.sleep(2)  # Show success message for 2 seconds
                    
                    # Clear all placeholders
                    info_placeholder.empty()
                    progress_placeholder.empty()
                    success_placeholder.empty()
                except Exception as e:
                    info_placeholder.error(f"Error downloading model: {e}")
                    raise
        
        # Load the model with a temporary message
        load_placeholder = st.empty()
        load_placeholder.info("Loading model...")
        model = tf.keras.models.load_model(model_path)
        # Clear the loading message
        load_placeholder.empty()
        
        # Get class indices from local file or download from Hugging Face
        if not os.path.exists(class_indices_path):
            # Create placeholder for class indices message
            indices_placeholder = st.empty()
            indices_placeholder.info("Downloading class indices from Hugging Face...")
            try:
                response = requests.get(class_indices_url)
                response.raise_for_status()
                class_indices = response.json()
                
                # Save class indices locally for future use
                with open(class_indices_path, "w") as f:
                    json.dump(class_indices, f)
                
                # Clear the message after successful download
                indices_placeholder.empty()
            except Exception as e:
                # Replace info with warning in case of error
                indices_placeholder.warning(f"Error downloading class indices: {e}")
                # Fall back to default class indices
                class_indices = {
                    "ABE": 0, "ART": 1, "BAS": 2, "BLA": 3, "EBO": 4, "EOS": 5, "FGC": 6,
                    "HAC": 7, "KSC": 8, "LYI": 9, "LYT": 10, "MMZ": 11, "MON": 12, "MYB": 13,
                    "NGB": 14, "NGS": 15, "NIF": 16, "OTH": 17, "PEB": 18, "PLM": 19, "PMO": 20
                }
                # Don't clear warning message as it contains useful error info
        else:
            # Load class indices from local file
            with open(class_indices_path, "r") as f:
                class_indices = json.load(f)
        
        # Invert dictionary to map from index to class name
        idx_to_class = {v: k for k, v in class_indices.items()}
        
        return model, idx_to_class
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("For demonstration purposes, we'll return a mock model and continue.")
        
        # Mock model and classes for UI demonstration
        from tensorflow.keras.applications import ResNet152
        mock_model = ResNet152(weights=None, include_top=True, classes=21)  # Updated to 21 classes
        
        # Updated mock class mapping to include all 21 classes
        mock_classes = {
            0: "ABE", 1: "ART", 2: "BAS", 3: "BLA", 4: "EBO", 5: "EOS", 6: "FGC",
            7: "HAC", 8: "KSC", 9: "LYI", 10: "LYT", 11: "MMZ", 12: "MON", 13: "MYB",
            14: "NGB", 15: "NGS", 16: "NIF", 17: "OTH", 18: "PEB", 19: "PLM", 20: "PMO"
        }
        
        return mock_model, mock_classes

# Function to preprocess image
def preprocess_image(img, target_size=(224, 224)):
    """Resize and preprocess image for ResNet152"""
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # ResNet152 preprocessing
    img_array = tf.keras.applications.resnet.preprocess_input(img_array)
    return img_array

# Function to make predictions
def predict_image(model, img_array, idx_to_class):
    """Predict class with confidence for the image"""
    predictions = model.predict(img_array)[0]
    top_3_idx = predictions.argsort()[-3:][::-1]
    top_3_classes = [idx_to_class[idx] for idx in top_3_idx]
    top_3_probs = [predictions[idx] for idx in top_3_idx]
    
    return list(zip(top_3_classes, top_3_probs))

# Function to display class descriptions
def get_class_description(class_name):
    """Return description for each class"""
    descriptions = {
        "ABE": "Abetalipoproteinemia - A rare inherited disorder that affects the absorption of dietary fats, cholesterol, and fat-soluble vitamins.",
        "ART": "Artifact - Abnormalities or structures in medical images that are not naturally present in the specimen being examined.",
        "BAS": "Basophilia - An increase in the number of basophils in the blood, often associated with allergic reactions or inflammatory conditions.",
        "BLA": "Blast Cells - Immature blood cells that are normally found in the bone marrow but not in peripheral blood, often indicative of leukemia.",
        "EBO": "Elliptocytosis & Ovalocytosis - Blood disorders characterized by abnormally shaped red blood cells that appear oval or elliptical.",
        "EOS": "Eosinophilia - An abnormally high concentration of eosinophils in the blood, often in response to allergic disorders or parasitic infections.",
        "FGC": "Familial Hypercholesterolemia - An inherited condition characterized by very high levels of cholesterol in the blood.",
        "HAC": "Hairy Cell Leukemia - A rare, slow-growing cancer of the blood in which the bone marrow makes too many B cells.",
        "KSC": "Kawasaki Syndrome - An acute febrile illness of unknown etiology that primarily affects children younger than 5 years of age.",
        "LYI": "Lymphocytic Infiltration - A condition characterized by the accumulation of lymphocytes in tissues.",
        "LYT": "Lymphocytes - A type of white blood cell that is crucial to the immune system's response to infection.",
        "MMZ": "Marginal Zone Lymphoma - A slow-growing B-cell non-Hodgkin lymphoma that arises from the marginal zone of the lymphoid tissue.",
        "MON": "Monocytes - The largest type of white blood cell, playing a key role in the immune system's response to inflammation and infection.",
        "MYB": "Myeloblasts - Immature cells that give rise to all the formed elements of the blood.",
        "NGB": "Neutrophilic Band Forms - Immature neutrophils characterized by a band-shaped nucleus, often seen in bacterial infections.",
        "NGS": "Neutrophilic Segmented - Mature neutrophils with segmented nuclei, the most common type of white blood cell involved in fighting bacterial infections.",
        "NIF": "Neutrophilic Infiltration - The accumulation of neutrophils in tissues, often as a response to acute inflammation or infection.",
        "OTH": "Other - Blood cell abnormalities that do not fit into the other defined categories.",
        "PEB": "Peripheral Erythroblastosis - The presence of nucleated red blood cells (erythroblasts) in peripheral blood, often indicating bone marrow stress.",
        "PLM": "Plasma Cells - Fully differentiated B cells that secrete large amounts of antibodies, their presence in peripheral blood can indicate multiple myeloma.",
        "PMO": "Promyelocytes - Early granulocyte precursors, their abnormal proliferation is characteristic of acute promyelocytic leukemia."
    }
    return descriptions.get(class_name, "No description available")

# Function to get sample image paths (you would need to provide these)
def get_sample_images():
    """Function to return sample images for demo purposes"""
    # In a real scenario, you'd have actual paths to sample images
    # This is a placeholder - you'll need to replace with actual images
    samples = {
        "Sample 1: Blood Cell Analysis": "path/to/sample1.jpg",
        "Sample 2: Tissue Section": "path/to/sample2.jpg",
        "Sample 3: Microscopy Image": "path/to/sample3.jpg",
    }
    return samples

# Function to create a PDF report (placeholder function)
def create_pdf_report(results):
    """Create a PDF report of the results - this is a placeholder function"""
    from datetime import datetime
    
    # In a real implementation, you would use a library like ReportLab to create a PDF
    # For now, we'll just create a text file as a demonstration
    buffer = io.BytesIO()
    
    report_text = f"""
MEDICAL IMAGE CLASSIFICATION REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PRIMARY DIAGNOSIS: {results[0][0]} ({results[0][1]*100:.1f}%)
{get_class_description(results[0][0])}

ALTERNATIVE DIAGNOSES:
"""
    
    for class_name, prob in results[1:]:
        report_text += f"- {class_name} ({prob*100:.1f}%): {get_class_description(class_name)}\n"
    
    report_text += """
DISCLAIMER:
This is an AI-assisted diagnosis tool for educational purposes only. 
Always consult with a qualified healthcare professional for medical advice and diagnosis.
"""
    
    buffer.write(report_text.encode())
    buffer.seek(0)
    return buffer

# Function to display classification results
def display_results(results):
    """Display the classification results in a formatted way"""
    # Create a modern card container for results
    st.markdown("""
    <div style="padding: 1rem; border-radius: 1rem; margin-top: 2rem; background: linear-gradient(to bottom right, #ffffff, #f8f9fa); 
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header' style='text-align: center; margin-bottom: 1.5rem;'>Classification Results</h2>", unsafe_allow_html=True)
    
    # Create a colorful card for primary diagnosis
    primary_class = results[0][0]
    primary_prob = results[0][1] * 100
    
    # Choose color based on confidence level
    color = "#4CAF50" if primary_prob > 90 else "#FF9800" if primary_prob > 70 else "#F44336"
    
    st.markdown(f"""
    <div style="background: linear-gradient(to right, {color}15, {color}05); 
        border-left: 4px solid {color}; padding: 1.5rem; border-radius: 0.5rem;">
        <h3 style="color: {color}; margin-top: 0;">Primary Diagnosis</h3>
        <p style="font-size: 1.5rem; font-weight: bold;">{primary_class} <span style="font-size: 1.2rem; color: {color};">({primary_prob:.1f}%)</span></p>
        <p>{get_class_description(primary_class)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display confidence scores
    st.markdown("<h3 style='margin-top: 2rem;'>Confidence Levels</h3>", unsafe_allow_html=True)
    
    # Display bar chart of top 3 predictions with improved styling
    fig, ax = plt.subplots(figsize=(10, 4))
    classes = [r[0] for r in results]
    scores = [r[1] * 100 for r in results]
    
    # Use a custom color palette for a modern look
    custom_palette = ["#4CAF50", "#2196F3", "#9C27B0"]
    # Updated to new Seaborn API to avoid deprecation warning
    bars = sns.barplot(x=scores, y=classes, hue=classes, palette=custom_palette, legend=False, ax=ax)
    
    # Style the chart
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", fontsize=11)
    ax.set_ylabel("Class", fontsize=11)
    ax.set_title("Top 3 Predictions", fontsize=13, fontweight="bold")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add percentage labels to the bars
    for i, score in enumerate(scores):
        ax.text(score + 1, i, f"{score:.1f}%", va='center', fontsize=11)
    
    st.pyplot(fig)
    
    # Additional information about other possible diagnoses
    st.markdown("<h3 style='margin-top: 1rem;'>Alternative Diagnoses</h3>", unsafe_allow_html=True)
    
    # Create columns for alternative diagnoses
    col1, col2 = st.columns(2)
    
    # Alternative diagnoses in a more compact format
    for i, (class_name, prob) in enumerate(results[1:]):
        with col1 if i == 0 else col2:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem;">
                <p style="font-weight: bold; margin-bottom: 0.2rem;">{class_name} ({prob*100:.1f}%)</p>
                <p style="font-size: 0.9rem; color: #666;">{get_class_description(class_name)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Add options to download or share results with better UI
    st.markdown("<h3 style='margin-top: 1.5rem;'>Report Options</h3>", unsafe_allow_html=True)
    
    # Store button states in session state to prevent page refresh issues
    if 'download_clicked' not in st.session_state:
        st.session_state.download_clicked = False
    if 'copy_clicked' not in st.session_state:
        st.session_state.copy_clicked = False
    
    # Create a key for this specific results set
    result_key = f"{primary_class}_{primary_prob:.1f}"
    download_key = f"download_{result_key}"
    copy_key = f"copy_{result_key}"
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create download button that doesn't refresh the page
        if st.download_button(
            label="📥 Download Report",
            data=create_pdf_report(results),
            file_name=f"medical_report_{primary_class}.txt",
            mime="text/plain",
            key=download_key
        ):
            st.session_state.download_clicked = True
    
    with col2:
        # Create button to copy results
        if st.button("📋 Copy Results", key=copy_key):
            st.session_state.copy_clicked = True
              # Show success messages if buttons were clicked
    if st.session_state.copy_clicked:        # Create a string with the results
        result_text = f"Primary Diagnosis: {results[0][0]} ({results[0][1]*100:.1f}%)\n"
        result_text += f"Description: {get_class_description(results[0][0])}\n\n"
        result_text += "Alternative Diagnoses:\n"
        for class_name, prob in results[1:]:
            result_text += f"- {class_name} ({prob*100:.1f}%)\n"
        
        # Use st.code which provides a built-in copy button
        st.markdown("### 📋 Copy Diagnosis Results")
        st.code(result_text, language=None)
        st.success("✅ Click the copy button in the top-right corner of the code block above to copy the results!")
        
        # Reset the flag after showing the message
        st.session_state.copy_clicked = False
    
    # Disclaimer with better styling
    st.markdown("""
    <div style="margin-top: 2rem; padding: 1rem; border-radius: 0.5rem; background-color: #E3F2FD; border-left: 4px solid #1976D2;">
        <p style="margin: 0; color: #0D47A1;"><strong>⚠️ IMPORTANT:</strong> This is an AI-assisted diagnosis tool for educational purposes only. Always consult with a qualified healthcare professional for medical advice and diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Close the card container
    st.markdown("</div>", unsafe_allow_html=True)

# Main app function
def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/microscope.png", width=100)
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses a ResNet152 deep learning model to classify medical images. "
        "Upload an image to get a prediction with confidence scores."
    )
    st.sidebar.title("Instructions")
    st.sidebar.markdown(
        """
        1. Upload a medical image using the file uploader
        2. Wait for the model to process the image
        3. View the classification results and confidence scores
        """
    )
    
    # Add model stats to sidebar
    st.sidebar.title("Model Performance")
    st.sidebar.markdown("""
        - **Training Accuracy**: ~95%
        - **Validation Accuracy**: ~92% 
        - **Test Accuracy**: ~91%
        
        *Note: Actual performance may vary based on image quality.*
    """)
    
    # Main content
    st.markdown("<h1 class='main-header'>Medical Image Classification System</h1>", unsafe_allow_html=True)
    
    # Introduction section
    st.markdown("<p class='info-text'>This system uses deep learning to classify medical images into different categories. "
                "The model was trained on a dataset containing images of various medical conditions.</p>", 
                unsafe_allow_html=True)
    
    # Create tabs for different functionality
    tab1, tab2 = st.tabs(["Image Classification", "About the Model"])
    
    with tab1:
        # Using a single column layout for vertical stacking
        st.markdown("<h2 class='sub-header'>Upload Image</h2>", unsafe_allow_html=True)
        # Create a modern card container for the image uploader
        st.markdown("""
        <div style="padding: 1rem; border-radius: 1rem; background: linear-gradient(to bottom right, #f8f9fa, #e9ecef); box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        
        # Add tabs for upload or sample
        upload_tab, sample_tab = st.tabs(["📤 Upload Your Image", "🔬 Use Sample Image"])
        
        with upload_tab:            # Track current upload with a session ID
            if 'current_upload_id' not in st.session_state:
                st.session_state.current_upload_id = ""
            
            # Use the file uploader
            uploaded_file = st.file_uploader("Choose a medical image file", 
                                          type=["jpg", "jpeg", "png"],
                                          help="Upload an image to classify")
            
            if uploaded_file is not None:
                # Generate a unique ID for this upload
                current_id = str(hash(uploaded_file.name + str(uploaded_file.size)))
                
                # Clear results if a new image is uploaded
                if current_id != st.session_state.current_upload_id:
                    st.session_state.classification_results = None
                    st.session_state.current_upload_id = current_id
                
                # Create columns for a better image layout
                img_col1, img_col2, img_col3 = st.columns([1, 3, 1])
                with img_col2:
                    # Display uploaded image with styling
                    img = Image.open(uploaded_file).convert('RGB')
                    st.image(img, caption="Uploaded Image", use_container_width=True)
                
                # Add a process button with better styling
                _, btn_col, _ = st.columns([1,1,1])
                with btn_col:
                    process_button = st.button("🔍 Classify Image", use_container_width=True)
                
                # Initialize session_state for saving results
                if 'classification_results' not in st.session_state:
                    st.session_state.classification_results = None
                
                if process_button:
                    with st.spinner("Processing image..."):
                        # Show a progress bar to indicate processing
                        progress_bar = st.progress(0)
                        for percent_complete in range(100):
                            # Simulate processing time
                            time.sleep(0.01)  # Fast enough not to be annoying
                            progress_bar.progress(percent_complete + 1)
                        
                        # Load model (handled by caching decorator)
                        model, idx_to_class = load_model()
                        
                        # Preprocess image
                        img_array = preprocess_image(img)
                        
                        # Make prediction
                        results = predict_image(model, img_array, idx_to_class)
                        
                        # Store results in session state
                        st.session_state.classification_results = results
                    
                    # Show success message
                    st.success("✅ Classification complete!")
                    
                    # Show results below the image
                    display_results(results)
                # Show previous results if they exist in session state
                elif st.session_state.classification_results is not None:
                    display_results(st.session_state.classification_results)
                    
        with sample_tab:
            # Get list of sample images from the Sample Images folder
            sample_images_path = os.path.join(os.path.dirname(__file__), "Sample Images")
            
            # Check if Sample Images folder exists
            if os.path.exists(sample_images_path) and os.path.isdir(sample_images_path):
                # Get all image files from the folder
                sample_files = [f for f in os.listdir(sample_images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                if sample_files:
                    # Create display options with class names
                    sample_options = [f"{f.split('.')[0]} - {get_class_description(f.split('.')[0])[:50]}..." for f in sample_files]
                    
                    # Sample image selection with better UI
                    st.markdown("### Select a sample image to test the model")
                    st.markdown("Choose from one of the 21 different medical image classes")
                    
                    sample_option = st.selectbox(
                        "Select a sample image class:", 
                        options=sample_options,
                        key="sample_selector"
                    )
                    
                    # Get the corresponding file from the selection
                    selected_index = sample_options.index(sample_option)
                    selected_file = sample_files[selected_index]
                    selected_class = selected_file.split('.')[0]
                    
                    # Load the selected image
                    img = None
                    try:
                        image_path = os.path.join(sample_images_path, selected_file)
                        img = Image.open(image_path).convert('RGB')
                        
                        # Create columns for a better image layout
                        img_col1, img_col2, img_col3 = st.columns([1, 3, 1])
                        with img_col2:
                            st.image(img, caption=f"Sample: {selected_class}", use_container_width=True)
                            st.markdown(f"**Class**: {selected_class}")
                            st.markdown(f"**Description**: {get_class_description(selected_class)}")
                    except Exception as e:
                        st.error(f"Error loading sample image: {str(e)}. Please try another image.")
                        img = None
                else:
                    st.warning("No sample images found in the Sample Images folder.")
                    img = None
            else:
                # Fallback to web images if Sample Images folder doesn't exist
                st.warning("Sample Images folder not found. Using web images instead.")
                
                # Web image samples as fallback
                web_samples = {
                    "ABE": "https://media.istockphoto.com/id/1311702896/photo/blood-cells-on-blue-background.jpg?s=612x612&w=0&k=20&c=CvMJ2YyWLC7FCbqYU_FrmNT9tnN5xY1MSLyWDJN6Yjg=",
                    "HAC": "https://media.istockphoto.com/id/629132908/photo/cancer-cell.jpg?s=612x612&w=0&k=20&c=Tn5IUQqZ6HZh0DO9KQLkcwOJ_ykMr4kkZlof5oTmNjE=",
                    "KSC": "https://media.istockphoto.com/id/851159150/photo/3d-bacteria-microbes.jpg?s=612x612&w=0&k=20&c=x7pzNeJagnviYuJgdxcjnIXlZjUJgANUjMB1gA02Vkg="
                }
                
                # Create options for web samples
                sample_options = [f"{k} - {get_class_description(k)[:50]}..." for k in web_samples.keys()]
                sample_option = st.selectbox("Select a sample:", options=sample_options)
                selected_class = sample_option.split(" - ")[0]
                
                # Try loading web image
                img = None
                try:
                    from urllib.request import urlopen
                    from io import BytesIO
                    
                    response = urlopen(web_samples[selected_class])
                    img_data = response.read()
                    img = Image.open(BytesIO(img_data)).convert('RGB')
                    
                    # Create columns for a better image layout
                    img_col1, img_col2, img_col3 = st.columns([1, 3, 1])
                    with img_col2:
                        st.image(img, caption=f"Sample: {selected_class}", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading sample image: {str(e)}. Please try uploading your own image.")
                    img = None
            
            # If we have a valid image, create a button to classify it
            if img is not None:
                _, btn_col, _ = st.columns([1,1,1])
                with btn_col:
                    process_sample_button = st.button("🔍 Analyze Sample", use_container_width=True)
                
                # Initialize session_state for sample results
                if 'sample_results' not in st.session_state:
                    st.session_state.sample_results = None
                
                if process_sample_button:
                    with st.spinner("Processing sample image..."):
                        # Show a progress bar
                        progress_bar = st.progress(0)
                        for percent_complete in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(percent_complete + 1)
                        
                        # Load model
                        model, idx_to_class = load_model()
                        
                        # Preprocess image
                        img_array = preprocess_image(img)
                        
                        # Make prediction
                        results = predict_image(model, img_array, idx_to_class)
                        
                        # Store in session state
                        st.session_state.sample_results = results
                    
                    # Show success message
                    st.success("✅ Analysis complete!")
                    
                    # Show results below the image
                    display_results(results)
                
                # Show previous results if they exist
                elif st.session_state.sample_results is not None:
                    display_results(st.session_state.sample_results)
        
        # Close the card container
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<h2 class='sub-header'>About the ResNet152 Model</h2>", unsafe_allow_html=True)
        st.markdown(
            """
            <p class='info-text'>This application uses a fine-tuned ResNet152 model that was trained on an augmented medical image dataset. 
            The model achieved high accuracy in distinguishing between 21 different medical conditions and blood cell types.</p>
            
            <p class='info-text'><b>Model Architecture:</b></p>
            - Based on ResNet152 pre-trained on ImageNet
            - Fine-tuned on medical images with data augmentation
            - Uses transfer learning for optimal performance
            - Trained with class balancing for improved accuracy
            
            <p class='info-text'><b>The 21 classes include:</b></p>
            """, 
            unsafe_allow_html=True
        )
        
        # Create expandable sections for all 21 class descriptions
        all_classes = [
            "ABE", "ART", "BAS", "BLA", "EBO", "EOS", "FGC", 
            "HAC", "KSC", "LYI", "LYT", "MMZ", "MON", "MYB", 
            "NGB", "NGS", "NIF", "OTH", "PEB", "PLM", "PMO"
        ]
        
        # Create a three-column layout for the class descriptions
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]
        
        # Distribute classes across the three columns
        for i, class_name in enumerate(all_classes):
            with columns[i % 3]:
                with st.expander(f"{class_name}"):
                    st.write(get_class_description(class_name))
        
        # Add technical details in an expander
        with st.expander("Technical Details"):
            st.markdown("""
                - **Framework**: TensorFlow/Keras
                - **Input Size**: 224x224 RGB images
                - **Base Model**: ResNet152 (pre-trained on ImageNet)
                - **Fine-tuning**: Last 30 layers fine-tuned
                - **Regularization**: Dropout (0.5) and L2 regularization
                - **Training**: SGD optimizer with momentum, learning rate decay
                - **Deployment**: Streamlit web application
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("<p class='info-text' style='text-align: center;'>© 2025 Medical Image Classification System</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
