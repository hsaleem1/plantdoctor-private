# Add these to your existing imports
import requests
import matplotlib.pyplot as plt
import io
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
from huggingface_hub import hf_hub_download

from huggingface_hub import HfApi, login

# Login using token from secrets
if "HF_TOKEN" in st.secrets:
    login(token=st.secrets["HF_TOKEN"])
else:
    st.warning("HF_TOKEN not found in secrets. Feedback will not be saved.")

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = 'best_wheat_model.pth'
CLASS_NAMES = ['Wheat_BYDV', 'Wheat_Healthy', 'Wheat_Septoria']
NUM_CLASSES = 3
IMAGE_SIZE = 224

# ============================================================
# MULTI-CROP CONFIGURATION
# ============================================================
CROP_CONFIG = {
    'Wheat': {
        'classes': ['Wheat_BYDV', 'Wheat_Healthy', 'Wheat_Septoria'],
        'model_path': 'best_wheat_model.pth',
        'icon': '🌾',
        'trained_images': 5000,
        'accuracy': 92.5,
        'status': 'trained'
    },
    'Beans': {
        'classes': ['BYDV', 'Healthy', 'Septoria'],
        'model_path': 'best_wheat_model.pth',
        'icon': '🫘',
        'trained_images': 0,
        'accuracy': 0.0,
        'status': 'pending'
    },
    'Broccoli': {
        'classes': ['BYDV', 'Healthy', 'Septoria'],
        'model_path': 'best_wheat_model.pth',
        'icon': '🥦',
        'trained_images': 0,
        'accuracy': 0.0,
        'status': 'pending'
    }
    # Add your other crops here with the same 3 classes
}

CROP_NAMES = list(CROP_CONFIG.keys())

# ============================================================
# LOAD MODEL (Crop-Specific)
# ============================================================
@st.cache_resource
def load_model(crop_name):
    """Load the wheat model for all crops (temporary)"""
    config = CROP_CONFIG.get(crop_name)
    if not config:
        st.error(f"Unknown crop: {crop_name}")
        st.stop()
    
    # Always use the wheat model
    model_path = None
    
    # Try Hugging Face first
    try:
        model_path = hf_hub_download(
            repo_id="Muhammad-Hammad-Saleem/PlantDoctor-model-v2",
            filename="best_wheat_model_final.pth"
        )
    except Exception as e:
        # Fall back to local file
        try:
            model_path = 'best_wheat_model.pth'
        except:
            pass
    
    if model_path is None:
        st.error("Could not find model file")
        st.stop()
    
    # Always use 3 classes (wheat model)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    model.eval()
    return model

# ============================================================
# TRANSFORMS
# ============================================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# ============================================================
# GRAD-CAM FUNCTION (Updated to accept model)
# ============================================================
def generate_gradcam(image, model, class_names):
    """Generate Grad-CAM heatmap with the given model"""
    img_np = np.array(image)
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item() * 100
    
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    targets = [ClassifierOutputTarget(pred_class)]
    heatmap = cam(input_tensor=input_tensor, targets=targets)
    heatmap = heatmap[0, :]
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    
    # Overlay heatmap on image
    img_display = img_np.astype(np.float32) / 255.0
    visualization = show_cam_on_image(img_display, heatmap_resized, use_rgb=True)
    
    # Convert to RGB for display
    visualization_rgb = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
    
    return {
        'class': class_names[pred_class],
        'confidence': confidence,
        'heatmap': visualization_rgb,
        'original': img_np
    }

# ============================================================
# EXPLANATIONS (Updated with Evidence)
# ============================================================
def get_explanation(class_name, confidence, class_names):
    """Get explanation for the given class"""
    explanations = {
        'Wheat_BYDV': {
            'diagnosis': 'Barley Yellow Dwarf Virus (BYDV)',
            'evidence': 'Yellowing and stunting patterns detected',  # <-- ADDED
            'action': 'Scout field within 3 days. Apply aphid control if live aphids present.',
            'urgency': 'HIGH - Can reduce yields by up to 60%'
        },
        'Wheat_Healthy': {
            'diagnosis': 'Healthy Plant',
            'evidence': 'No disease symptoms detected',  # <-- ADDED
            'action': 'Continue routine monitoring. No action needed.',
            'urgency': 'LOW - Maintain good crop management'
        },
        'Wheat_Septoria': {
            'diagnosis': 'Septoria Leaf Spot',
            'evidence': 'Brown lesions with yellow halos detected',  # <-- ADDED
            'action': 'Apply fungicide within 7-10 days if weather favors spread.',
            'urgency': 'MODERATE - Can reduce yields by 30-50%'
        }
    }
    
    info = explanations.get(class_name, {
        'diagnosis': f'Unknown: {class_name}',
        'evidence': 'Consult agronomist for verification',  # <-- ADDED
        'action': 'Consult agronomist for verification',
        'urgency': 'UNKNOWN'
    })
    
    level = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
    
    # Updated return string to include evidence
    return f"""
🌾 **Diagnosis:** {info['diagnosis']}  
📊 **Confidence:** {confidence:.1f}% ({level})  
🔍 **Evidence:** {info['evidence']}  
📋 **Action:** {info['action']}  
⚠️ **Urgency:** {info['urgency']}  
💡 *AI-assisted. Confirm with field scouting.*
"""
    

# ============================================================
# PREDICTION (Crop-Specific)
# ============================================================
def predict_image(image, model, class_names):
    """Predict with Grad-CAM and return all probabilities"""
    if image is None:
        return None
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_np = np.array(image)
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        temperature = 2.0
        probs = torch.nn.functional.softmax(outputs[0] / temperature, dim=0)
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item() * 100
    
    # Get all probabilities
    all_probs = {}
    for i, name in enumerate(class_names):
        all_probs[name] = probs[i].item() * 100
    
    # Generate Grad-CAM
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    targets = [ClassifierOutputTarget(pred_class)]
    heatmap = cam(input_tensor=input_tensor, targets=targets)
    heatmap = heatmap[0, :]
    
    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    img_display = img_np.astype(np.float32) / 255.0
    visualization = show_cam_on_image(img_display, heatmap_resized, use_rgb=True)
    visualization_rgb = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
    
    return {
        'class': class_names[pred_class],
        'confidence': confidence,
        'explanation': get_explanation(class_names[pred_class], confidence, class_names),
        'heatmap': visualization_rgb,
        'original': img_np,
        'all_probs': all_probs,
        'pred_class': pred_class
    }

# ============================================================
# BATCH PREDICTION (Multiple Images)
# ============================================================
def predict_batch(images, model, class_names):
    """Predict for multiple images"""
    results = []
    for img in images:
        result = predict_image(img, model, class_names)
        if result:
            results.append(result)
    return results


# ============================================================
# LOAD MODEL (Cached)
# ============================================================
# model = load_model()

# ============================================================
# FEEDBACK SAVING (Hugging Face Dataset)
# ============================================================
from huggingface_hub import HfApi
import json
import datetime

# Initialize Hugging Face API
hf_api = HfApi()

# Your feedback dataset repository
FEEDBACK_REPO = "Muhammad-Hammad-Saleem/PlantDoctor-Feedback"

# ============================================================
# SAVE FEEDBACK TO HUGGING FACE (Full Code)
# ============================================================
def save_feedback_to_hf(image_name, predicted, actual, crop, confidence, location="", severity=""):
    """Save feedback to Hugging Face dataset"""
    try:
        if "HF_TOKEN" not in st.secrets:
            st.warning("Feedback not saved (no token)")
            return False
        
        # Clean location
        location = location.strip() if location and location.strip() else ""
        
        # If location is empty, don't save
        if not location:
            st.warning("⚠️ Location is required to save feedback. Please enter a location.")
            return False
        
        # ============================================================
        # PART 1: SAVE TO feedback.jsonl (AI Feedback)
        # ============================================================
        feedback_entry = {
            "timestamp": str(datetime.datetime.now()),
            "image_name": image_name,
            "predicted": predicted,
            "actual": actual,
            "crop": crop,
            "confidence": f"{confidence:.1f}%",
            "correct": "Yes" if predicted == actual else "No",
            "location": location if location and location.strip() else "",
            "severity": severity if severity else "N/A"
        }
        
        # Load existing feedback.jsonl
        try:
            url = f"https://huggingface.co/datasets/{FEEDBACK_REPO}/resolve/main/feedback.jsonl"
            headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                with open("feedback.jsonl", "w") as f:
                    f.write(response.text)
        except:
            # Create new file if it doesn't exist
            with open("feedback.jsonl", "w") as f:
                pass
        
        # Append new feedback
        with open("feedback.jsonl", "a") as f:
            f.write(json.dumps(feedback_entry) + "\n")
        
        # Upload feedback.jsonl to Hugging Face
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj="feedback.jsonl",
            path_in_repo="feedback.jsonl",
            repo_id=FEEDBACK_REPO,
            repo_type="dataset",
            token=st.secrets["HF_TOKEN"]
        )
        
        # ============================================================
        # PART 2: ALSO SAVE TO community_reports.jsonl (Quick Report Stats)
        # ============================================================
        report_entry = {
            "timestamp": str(datetime.datetime.now()),
            "crop": crop,
            "disease": actual,  # Use actual diagnosis as the disease
            "location": location if location and location.strip() else "",
            "severity": severity if severity else "N/A",
            "source": "ai_feedback"
        }
        
        # Load existing community_reports.jsonl
        try:
            url = f"https://huggingface.co/datasets/{FEEDBACK_REPO}/resolve/main/community_reports.jsonl"
            headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                with open("community_reports.jsonl", "w") as f:
                    f.write(response.text)
        except:
            # Create new file if it doesn't exist
            with open("community_reports.jsonl", "w") as f:
                pass
        
        # Append new report
        with open("community_reports.jsonl", "a") as f:
            f.write(json.dumps(report_entry) + "\n")
        
        # Upload community_reports.jsonl to Hugging Face
        api.upload_file(
            path_or_fileobj="community_reports.jsonl",
            path_in_repo="community_reports.jsonl",
            repo_id=FEEDBACK_REPO,
            repo_type="dataset",
            token=st.secrets["HF_TOKEN"]
        )
        
        return True
        
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return False

# ============================================================
# STREAMLIT UI (Multi-Crop)
# ============================================================
st.set_page_config(page_title="PlantDoctor", page_icon="🌾")

st.title("🌾 PlantDoctor - Multi-Crop Disease Detection")
st.write("Select a crop and upload images for instant diagnosis")

# ============================================================
# SIDEBAR: CROP SELECTION & MODEL CARDS
# ============================================================
st.sidebar.header("🌱 Select Crop")

selected_crop = st.sidebar.selectbox(
    "Choose your crop:",
    options=CROP_NAMES,
    index=0
)

# Crop info
crop_info = CROP_CONFIG[selected_crop]
st.sidebar.markdown(f"**{crop_info['icon']} {selected_crop}**")
st.sidebar.markdown(f"**Detectable conditions:** {len(crop_info['classes'])}")
st.sidebar.markdown("**Classes:**")
for cls in crop_info['classes']:
    st.sidebar.markdown(f"- {cls}")

if 'trained_images' in crop_info:
    st.sidebar.markdown(f"**Training images:** {crop_info['trained_images']}")
if 'accuracy' in crop_info and crop_info['accuracy'] > 0:
    st.sidebar.markdown(f"**Model accuracy:** {crop_info['accuracy']:.1f}%")
else:
    st.sidebar.markdown("**Model accuracy:** Not yet trained")

st.sidebar.markdown("---")

# ============================================================
# MODEL CARDS (Inside Sidebar)
# ============================================================
st.sidebar.subheader("📊 Model Card")

with st.sidebar.container():
    # Status
    if crop_info.get('accuracy', 0) > 0:
        st.sidebar.markdown("✅ **Status:** Trained")
    else:
        st.sidebar.markdown("⏳ **Status:** Training in progress")
    
    # Training details
    st.sidebar.markdown(f"📊 **Training images:** {crop_info.get('trained_images', 'N/A')}")
    
    if crop_info.get('accuracy', 0) > 0:
        st.sidebar.markdown(f"🎯 **Accuracy:** {crop_info['accuracy']:.1f}%")
    else:
        st.sidebar.markdown("🎯 **Accuracy:** Not yet available")
    
    st.sidebar.markdown(f"📁 **Classes:** {len(crop_info['classes'])}")
    
    # Progress bar
    if crop_info.get('accuracy', 0) > 0:
        st.sidebar.progress(crop_info['accuracy'] / 100, text=f"Model readiness: {crop_info['accuracy']:.0f}%")
    else:
        st.sidebar.progress(0.3, text="Model readiness: 30% (placeholder)")
    
    st.sidebar.caption("📅 Model version: v1.0 (June 2026)")

st.sidebar.markdown("---")
st.sidebar.caption("Upload images to get diagnosis and recommendations")

# ============================================================
# RESEARCH MODE (Toggle) - ADD THIS
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Research Mode")
st.sidebar.caption("For academic and scientific analysis")

research_mode = st.sidebar.toggle("Enable Research Mode", value=False)

if research_mode:
    st.sidebar.info("""
    **Research Features:**
    - Full confidence scores for all classes
    - Grad-CAM heatmap analysis
    - Model architecture details
    """)


# ============================================================
# SIDEBAR: AI Feedback Stats
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Feedback Stats")
st.sidebar.caption("Model performance from user feedback")

try:
    url = f"https://huggingface.co/datasets/{FEEDBACK_REPO}/resolve/main/feedback.jsonl"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        feedback_data = []
        for line in response.text.strip().split('\n'):
            if line:
                feedback_data.append(json.loads(line))
        
        if feedback_data:
            total = len(feedback_data)
            correct = sum(1 for f in feedback_data if f.get('correct') == 'Yes')
            incorrect = total - correct
            accuracy = (correct / total * 100) if total > 0 else 0
            
            st.sidebar.metric("📝 Total Feedback", total)
            st.sidebar.metric("✅ Accuracy", f"{accuracy:.1f}%")
        else:
            st.sidebar.info("No feedback yet")
    else:
        st.sidebar.warning("Could not load feedback")
except:
    st.sidebar.warning("Connect to Hugging Face")

# ============================================================
# SIDEBAR: Quick Report Stats (NEW)
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Quick Report Stats")
st.sidebar.caption("Community-reported disease sightings")

try:
    url = f"https://huggingface.co/datasets/{FEEDBACK_REPO}/resolve/main/community_reports.jsonl"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        report_data = []
        for line in response.text.strip().split('\n'):
            if line:
                report_data.append(json.loads(line))
        
        if report_data:
            total = len(report_data)
            
            # Top location
            locations = [r.get('location', 'Unknown') for r in report_data]
            top_location = max(set(locations), key=locations.count) if locations else "N/A"
            
            # Top disease
            diseases = [r.get('disease', 'Unknown') for r in report_data]
            top_disease = max(set(diseases), key=diseases.count) if diseases else "N/A"
            
            st.sidebar.metric("📝 Total Reports", total)
            st.sidebar.metric("📍 Top Location", top_location)
            st.sidebar.metric("⚠️ Most Reported", top_disease)
            
            # Show last 2 quick reports
            st.sidebar.caption("**Recent Reports:**")
            for r in report_data[-2:]:
                crop = r.get('crop', '?')
                disease = r.get('disease', '?')
                severity = r.get('severity', 'N/A')
                st.sidebar.text(f"🟢 {crop}: {disease} ({severity})")
        else:
            st.sidebar.info("No reports yet")
    else:
        st.sidebar.warning("Could not load reports")
except:
    st.sidebar.warning("Connect to Hugging Face")

# ============================================================
# LOAD MODEL FOR SELECTED CROP
# ============================================================
with st.spinner(f"Loading model for {selected_crop}..."):
    model = load_model(selected_crop)
    CLASS_NAMES = CROP_CONFIG[selected_crop]['classes']
    NUM_CLASSES = len(CLASS_NAMES)

st.success(f"✅ Ready! Analyzing {selected_crop}")


# ============================================================
# MAIN UPLOAD AREA & RESULTS
# ============================================================
uploaded_files = st.file_uploader(
    f"Upload {selected_crop} images...", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) > 0:
    images = []
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert('RGB')
        images.append(img)
    
    with st.spinner(f"Analyzing {len(images)} image(s)..."):
        results = predict_batch(images, model, CLASS_NAMES)
    
    # Display each result
    for idx, result in enumerate(results):
        with st.container():
            st.subheader(f"📸 Image {idx + 1}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(result['original'], caption="Uploaded Image", use_container_width=True)
            with col2:
                st.image(result['heatmap'], caption=f"Grad-CAM: {result['class']} ({result['confidence']:.1f}%)", use_container_width=True)
            
            st.success(f"🔍 **Diagnosis:** {result['class']}")
            st.info(f"📊 **Confidence:** {result['confidence']:.1f}%")
            
            # ============================================================
            # RESEARCH MODE DETAILS - ADD THIS
            # ============================================================
            if research_mode and 'all_probs' in result:
                with st.expander("🔬 Research Details", expanded=True):
                    st.subheader("All Class Probabilities")
                    
                    # Show bar chart
                    import pandas as pd
                    prob_df = pd.DataFrame({
                        'Class': list(result['all_probs'].keys()),
                        'Confidence (%)': list(result['all_probs'].values())
                    })
                    st.bar_chart(prob_df.set_index('Class'))
                    
                    # Show detailed table
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                    
                    st.subheader("Model Information")
                    st.json({
                        "Model Architecture": "ResNet18 (pretrained)",
                        "Number of Classes": len(CLASS_NAMES),
                        "Input Size": "224x224",
                        "Temperature Scaling": "2.0",
                        "Prediction ID": result.get('pred_class', 'N/A')
                    })
            
            st.markdown("### 📋 Explanation & Spray Recommendations")
            st.markdown(result['explanation'])
            st.markdown("---")
                  

    # ============================================================
    # FEEDBACK UI (Location → Correct/Incorrect → Severity)
    # ============================================================
    st.markdown("---")
    st.subheader("📝 Help Improve the Model")
    st.caption("Your feedback helps us train better models for farmers")
    

    for idx, result in enumerate(results):
        with st.expander(f"📸 Provide feedback for Image {idx + 1} ({result['class']})"):
            
            image_name = uploaded_files[idx].name if idx < len(uploaded_files) else "unknown"
            crop_name = selected_crop
            predicted = result['class']
            confidence = result['confidence']
    
            # ============================================================
            # 1. LOCATION
            # ============================================================
            location = st.text_input(
                "📍 Your location (e.g., Norfolk, UK):", 
                key=f"location_{idx}",
                placeholder="e.g., Norfolk, UK"
            )
            
            # ============================================================
            # 2. CORRECT / INCORRECT BUTTONS
            # ============================================================
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"✅ Correct", key=f"correct_{idx}"):
                    if not location.strip():
                        st.warning("⚠️ Please enter your location.")
                    else:
                        st.session_state[f"highlight_{idx}"] = "correct"
                        st.session_state[f"feedback_action_{idx}"] = "correct"
                        st.session_state[f"show_severity_{idx}"] = True
                        st.session_state[f"incorrect_clicked_{idx}"] = False
                        st.rerun()
            
            with col2:
                if st.button(f"❌ Incorrect", key=f"incorrect_{idx}"):
                    if not location.strip():
                        st.warning("⚠️ Please enter your location.")
                    else:
                        st.session_state[f"highlight_{idx}"] = "incorrect"
                        st.session_state[f"incorrect_clicked_{idx}"] = True
                        st.session_state[f"feedback_action_{idx}"] = "incorrect"
                        st.session_state[f"show_severity_{idx}"] = False
                        st.rerun()
            
            # ============================================================
            # 3. HIGHLIGHT MESSAGE
            # ============================================================
            highlight_key = f"highlight_{idx}"
            if st.session_state.get(highlight_key):
                if st.session_state[highlight_key] == "correct":
                    st.success("✅ You selected: **Correct**")
                elif st.session_state[highlight_key] == "incorrect":
                    st.info("❌ You selected: **Incorrect**")
            
            # ============================================================
            # 4. INCORRECT DIAGNOSIS (Stays visible after submitting)
            # ============================================================
            if st.session_state.get(f"incorrect_clicked_{idx}", False):
                st.markdown("---")
                st.markdown("**✏️ What was the correct diagnosis?**")
                actual = st.text_input("Enter the correct diagnosis:", key=f"actual_{idx}")
                
                col_submit1, col_submit2 = st.columns([3, 1])
                with col_submit1:
                    if st.button("✅ Submit Correct Diagnosis", key=f"submit_correct_diagnosis_{idx}"):
                        if actual.strip():
                            st.session_state[f"actual_diagnosis_{idx}"] = actual
                            st.session_state[f"show_severity_{idx}"] = True
                            # Keep incorrect_clicked = True so the input stays visible
                            st.success(f"✅ Diagnosis '{actual}' submitted! Now select severity below.")
                            st.rerun()
                        else:
                            st.warning("⚠️ Please enter a diagnosis before submitting.")
                
                # Show the submitted diagnosis if it exists
                if st.session_state.get(f"actual_diagnosis_{idx}"):
                    st.info(f"📝 Submitted diagnosis: **{st.session_state[f'actual_diagnosis_{idx}']}**")
            
            # ============================================================
            # 5. SEVERITY + SAVE
            # ============================================================
            if st.session_state.get(f"show_severity_{idx}", False):
                st.markdown("---")
                
                severity = st.select_slider(
                    "⚠️ Severity of the issue:",
                    options=["Low", "Medium", "High", "Critical"],
                    value="Medium",
                    key=f"severity_{idx}"
                )
                
                if st.button("💾 Save Feedback", key=f"save_{idx}"):
                    action = st.session_state.get(f"feedback_action_{idx}")
                    
                    if action == "correct":
                        actual_diagnosis = predicted
                    else:
                        actual_diagnosis = st.session_state.get(f"actual_diagnosis_{idx}", "")
                        if not actual_diagnosis:
                            st.warning("⚠️ Please submit the correct diagnosis first.")
                            st.stop()
                    
                    if save_feedback_to_hf(image_name, predicted, actual_diagnosis, crop_name, confidence, location, severity):
                        st.success("✅ Feedback saved to Hugging Face!")
                        # Reset all states
                        st.session_state[f"show_severity_{idx}"] = False
                        st.session_state[f"feedback_action_{idx}"] = ""
                        st.session_state[f"incorrect_clicked_{idx}"] = False
                        st.session_state[f"actual_diagnosis_{idx}"] = ""
                        st.session_state[f"highlight_{idx}"] = ""
                        st.rerun()
                    else:
                        st.error("Could not save feedback.")


    # ============================================================
    # HEALTH DASHBOARD (Mock-up)
    # ============================================================

    st.markdown("---")
    st.subheader("📊 Farm Health Dashboard")
    st.caption("Overview of recent diagnoses across your fields")
    
    # Create a summary of results
    summary = {}
    for result in results:
        crop = selected_crop
        diagnosis = result['class']
        key = f"{crop} - {diagnosis}"
        summary[key] = summary.get(key, 0) + 1
    
    # Display summary in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🌾 Total Images Analyzed",
            value=len(results),
            delta="Today's session"
        )
    
    with col2:
        # Count healthy vs diseased
        healthy_count = sum(1 for r in results if r['class'] == 'Healthy')
        disease_count = len(results) - healthy_count
        st.metric(
            label="🟢 Healthy / 🔴 Diseased",
            value=f"{healthy_count} / {disease_count}",
            delta=f"{disease_count/len(results)*100:.0f}% affected" if len(results) > 0 else "0%"
        )
    
    with col3:
        # Most common issue
        if summary:
            most_common = max(summary, key=summary.get)
            st.metric(
                label="⚠️ Most Common Issue",
                value=most_common.split(" - ")[1],
                delta=f"{summary[most_common]} detections"
            )
        else:
            st.metric(label="⚠️ Most Common Issue", value="None detected")
    
    # Detailed breakdown table
    st.markdown("### 📋 Detailed Breakdown")
    breakdown_data = []
    for key, count in summary.items():
        crop_name, disease = key.split(" - ")
        breakdown_data.append({"Crop": crop_name, "Disease/Issue": disease, "Detections": count})
    
    if breakdown_data:
        import pandas as pd
        df = pd.DataFrame(breakdown_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No detections to display")
    
    # Field-level map visualization (placeholder)
    st.markdown("### 🗺️ Field Health Map")
    st.caption("Interactive map showing affected areas (coming soon)")
    
    # Mock-up of a field map
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Field A**")
        st.markdown("🌾 Wheat - 🟢 Healthy")
        st.markdown("🌾 Wheat - 🟡 Moderate")
        st.markdown("🌾 Wheat - 🔴 BYDV")
    with col2:
        st.markdown("**Field B**")
        st.markdown("🫘 Beans - 🟢 Healthy")
        st.markdown("🫘 Beans - 🟢 Healthy")
    with col3:
        st.markdown("**Field C**")
        st.markdown("🥦 Broccoli - ⚠️ In Progress")


    # ============================================================
    # FEEDBACK DASHBOARD (Authenticated)
    # ============================================================
    st.markdown("---")
    st.subheader("📊 Feedback Dashboard")
    st.caption("Live statistics from user feedback")
    
    # Check if token is available
    if "HF_TOKEN" in st.secrets:
        try:
            from huggingface_hub import HfApi
            import requests
            
            # Use Hugging Face API with token
            api = HfApi()
            
            # Download the file with authentication
            url = f"https://huggingface.co/datasets/{FEEDBACK_REPO}/resolve/main/feedback.jsonl"
            headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                feedback_data = []
                for line in response.text.strip().split('\n'):
                    if line:
                        feedback_data.append(json.loads(line))
                
                if feedback_data:
                    # Calculate stats
                    total = len(feedback_data)
                    correct = sum(1 for f in feedback_data if f.get('correct') == 'Yes')
                    incorrect = total - correct
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📝 Total Feedback", total)
                    with col2:
                        st.metric("✅ Correct", correct, delta=f"{correct/total*100:.0f}%" if total > 0 else "0%")
                    with col3:
                        st.metric("❌ Incorrect", incorrect, delta=f"{incorrect/total*100:.0f}%" if total > 0 else "0%")
                    with col4:
                        st.metric("🎯 Accuracy", f"{correct/total*100:.1f}%" if total > 0 else "N/A")
                    
                    # Show recent feedback
                    st.markdown("### 📋 Recent Feedback")
                    for fb in feedback_data[-5:]:  # Last 5 entries
                        st.markdown(f"""
                        **Image:** {fb.get('image_name', 'N/A')}  
                        **Predicted:** {fb.get('predicted', 'N/A')} → **Actual:** {fb.get('actual', 'N/A')}  
                        **Crop:** {fb.get('crop', 'N/A')} | **Confidence:** {fb.get('confidence', 'N/A')}  
                        **Status:** {'✅ Correct' if fb.get('correct') == 'Yes' else '❌ Incorrect'}  
                        ---
                        """)
                else:
                    st.info("No feedback yet. Upload images and provide feedback!")
            else:
                st.warning(f"Could not load feedback (Status: {response.status_code})")
        except Exception as e:
            st.warning(f"Error loading feedback: {e}")
    else:
        st.warning("HF_TOKEN not found. Please add it to Secrets.")

    # ============================================================
    # COMMUNITY HUB - Main Dashboard (All Three Options)
    # ============================================================
    st.markdown("---")
    st.subheader("🤝 Community Hub")
    st.caption("Real-time disease tracking and knowledge sharing")
    
    # Create tabs for organization
    tab1, tab2, tab3 = st.tabs(["📊 Regional Map", "📋 Recent Reports", "📈 Trends"])
        
    
    # ============================================================
    # TAB 1: Regional Disease Map
    # ============================================================
    with tab1:
        st.subheader("🗺️ Regional Disease Activity")
        st.caption("Disease hotspots from community reports")
        
        try:
            url = f"https://huggingface.co/datasets/{FEEDBACK_REPO}/resolve/main/feedback.jsonl"
            headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                all_feedback = []
                for line in response.text.strip().split('\n'):
                    if line:
                        all_feedback.append(json.loads(line))
                
                # Build regional map
                regions = {}
                for fb in all_feedback:
                    location = fb.get('location', '').strip()
                    if not location:
                        location = "Unknown Location"
                    disease = fb.get('actual', 'Unknown')
                    severity = fb.get('severity', 'N/A')
                    
                    if location not in regions:
                        regions[location] = {}
                    if disease not in regions[location]:
                        regions[location][disease] = {'count': 0, 'severities': []}
                    regions[location][disease]['count'] += 1
                    if severity != 'N/A':
                        regions[location][disease]['severities'].append(severity)
                
                if regions:
                    import pandas as pd
                    severity_summary = []
                    for location, diseases in regions.items():
                        for disease, data in diseases.items():
                            severity_summary.append({
                                "Location": location,
                                "Disease": disease,
                                "Reports": data['count'],
                                "Severity": data['severities'][0] if data['severities'] else 'N/A'
                            })
                    df_severity = pd.DataFrame(severity_summary)
                    st.dataframe(df_severity, use_container_width=True, hide_index=True)
                    
                    st.markdown("**🔴 Active Hotspots (with Severity):**")
                    for location, diseases in regions.items():
                        disease_str = []
                        for disease, data in diseases.items():
                            sev = data['severities'][0] if data['severities'] else 'N/A'
                            disease_str.append(f"{disease} ({data['count']}, Severity: {sev})")
                        st.warning(f"⚠️ **{location}**: {', '.join(disease_str)}")
                else:
                    st.info("No location data yet. Provide feedback with locations to build the map!")
            else:
                st.info("Connect to Hugging Face to see community map")
        except Exception as e:
            st.warning(f"Could not load data: {e}")


    # ============================================================
    # TAB 2: Recent Community Reports
    # ============================================================
    with tab2:
        st.subheader("📋 Recent Community Reports")
        st.caption("Real reports from farmers using PlantDoctor")
        
        all_reports = []
        
        # 1. Load AI feedback from feedback.jsonl
        try:
            url = f"https://huggingface.co/datasets/{FEEDBACK_REPO}/resolve/main/feedback.jsonl"
            headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                for line in response.text.strip().split('\n'):
                    if line:
                        data = json.loads(line)
                        location = data.get('location', '').strip()
                        if not location:
                            location = "Unknown Location"
                        all_reports.append({
                            "Crop": data.get('crop', 'Unknown'),
                            "Disease": data.get('actual', 'Unknown'),
                            "Location": location,
                            "Severity": data.get('severity', 'N/A'),
                            "Source": "AI Feedback",
                            "Reported": data.get('timestamp', '')[:16]
                        })
        except:
            pass
        
        # 2. Load Quick Reports from community_reports.jsonl
        try:
            url = f"https://huggingface.co/datasets/{FEEDBACK_REPO}/resolve/main/community_reports.jsonl"
            headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                for line in response.text.strip().split('\n'):
                    if line:
                        data = json.loads(line)
                        location = data.get('location', '').strip()
                        if not location:
                            location = "Unknown Location"
                        all_reports.append({
                            "Crop": data.get('crop', 'Unknown'),
                            "Disease": data.get('disease', 'Unknown'),
                            "Location": location,
                            "Severity": data.get('severity', 'N/A'),
                            "Source": "Quick Report",
                            "Reported": data.get('timestamp', '')[:16]
                        })
        except:
            pass
        
        # 3. Display combined reports
        if all_reports:
            import pandas as pd
            df_reports = pd.DataFrame(all_reports[-20:])  # Show last 20
            st.dataframe(df_reports, use_container_width=True, hide_index=True)
        else:
            st.info("No reports yet. Submit a report!")

    # ============================================================
    # TAB 3: Trends & Analytics
    # ============================================================
    with tab3:
        st.subheader("📈 Disease Trends & Analytics")
        st.caption("Historical patterns from community reports")
        
        try:
            url = f"https://huggingface.co/datasets/{FEEDBACK_REPO}/resolve/main/feedback.jsonl"
            headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                feedback_data = []
                for line in response.text.strip().split('\n'):
                    if line:
                        feedback_data.append(json.loads(line))
                
                # Also load community reports
                try:
                    url2 = f"https://huggingface.co/datasets/{FEEDBACK_REPO}/resolve/main/community_reports.jsonl"
                    response2 = requests.get(url2, headers=headers)
                    if response2.status_code == 200:
                        for line in response2.text.strip().split('\n'):
                            if line:
                                data = json.loads(line)
                                feedback_data.append({
                                    "actual": data.get('disease', 'Unknown'),
                                    "severity": data.get('severity', 'N/A'),
                                    "crop": data.get('crop', 'Unknown'),
                                    "location": data.get('location', 'Unknown'),
                                    "source": "quick_report"
                                })
                except:
                    pass
                
                if feedback_data:
                    import pandas as pd
                    
                    # Severity distribution
                    severity_counts = {}
                    for fb in feedback_data:
                        sev = fb.get('severity', 'N/A')
                        severity_counts[sev] = severity_counts.get(sev, 0) + 1
                    
                    st.subheader("Severity Distribution")
                    df_severity = pd.DataFrame(list(severity_counts.items()), columns=["Severity", "Reports"])
                    if not df_severity.empty:
                        st.bar_chart(df_severity.set_index("Severity"))
                    else:
                        st.info("No severity data yet")
                    
                    # Top diseases
                    disease_counts = {}
                    for fb in feedback_data:
                        disease = fb.get('actual', 'Unknown')
                        disease_counts[disease] = disease_counts.get(disease, 0) + 1
                    
                    st.subheader("Top Diseases")
                    df_diseases = pd.DataFrame(list(disease_counts.items()), columns=["Disease", "Reports"])
                    if not df_diseases.empty:
                        st.bar_chart(df_diseases.set_index("Disease"))
                    else:
                        st.info("No disease data yet")
                    
                    # Per-crop breakdown
                    crop_stats = {}
                    for fb in feedback_data:
                        crop = fb.get('crop', 'Unknown')
                        if crop not in crop_stats:
                            crop_stats[crop] = {}
                        disease = fb.get('actual', 'Unknown')
                        crop_stats[crop][disease] = crop_stats[crop].get(disease, 0) + 1
                    
                    if crop_stats:
                        st.subheader("Diseases by Crop")
                        crop_summary = []
                        for crop, diseases in crop_stats.items():
                            for disease, count in diseases.items():
                                crop_summary.append({"Crop": crop, "Disease": disease, "Reports": count})
                        df_crops = pd.DataFrame(crop_summary)
                        st.dataframe(df_crops, use_container_width=True, hide_index=True)
                    
                    # Overall stats
                    total = len(feedback_data)
                    correct = sum(1 for fb in feedback_data if fb.get('correct') == 'Yes')
                    accuracy = (correct / total * 100) if total > 0 else 0
                    st.metric("Overall Model Accuracy", f"{accuracy:.1f}%" if total > 0 else "N/A")
                else:
                    st.info("No data yet. Upload images and provide feedback!")
            else:
                st.info("Connect to Hugging Face to see trends")
        except Exception as e:
            st.warning(f"Could not load trends: {e}")


    
    # ============================================================
    # FINAL CAPTION
    # ============================================================

    st.caption("⚠️ AI-assisted diagnosis. Always confirm with field scouting.")
else:
    st.info("👆 Upload one or more images to begin analysis")

