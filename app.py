import streamlit as st
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
import torch
from torchvision import transforms
import cv2  # Minimal use for I/O
from skimage.transform import PiecewiseAffineTransform, warp

# Initialize InsightFace
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Simulated AI model for skin enhancement
class SkinEnhancer:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.model = lambda x: x  # Placeholder

    def enhance(self, image):
        img_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            enhanced = self.model(img_tensor)
        enhanced = enhanced.squeeze(0).permute(1, 2, 0).numpy()
        enhanced = (enhanced * 0.5 + 0.5) * 255
        return enhanced.astype(np.uint8)

skin_enhancer = SkinEnhancer()

def apply_face_shape(image, shape_factor=0.1):
    faces = face_app.get(image)
    if not faces:
        return image
    
    face = faces[0]
    landmarks = face.landmark_2d_106
    h, w = image.shape[:2]
    
    src_pts = landmarks.astype(np.float32)
    dst_pts = src_pts.copy()
    
    center_x = np.mean(src_pts[:, 0])
    for i in range(len(src_pts)):
        x, y = src_pts[i]
        dx = center_x - x
        dist = abs(dx)
        if dist > 0 and (i < 34 or (34 <= i < 50)):
            shift = shape_factor * dist * (0.8 if i < 34 else 0.4)
            new_x = x + (dx * shift / dist)
            dst_pts[i] = [new_x, y]
    
    tform = PiecewiseAffineTransform()
    tform.estimate(src_pts, dst_pts)
    warped = warp(image, tform, output_shape=(h, w))
    warped = (warped * 255).astype(np.uint8)
    
    return warped

def auto_adjust_parameters(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    mean_brightness = np.mean(v_channel) / 255.0
    
    auto_brightness = 1.0
    if mean_brightness < 0.4:
        auto_brightness = 1.5
    elif mean_brightness > 0.7:
        auto_brightness = 0.8
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = gray.std() / 255.0
    auto_smooth = 1.0 + (contrast * 2.0)
    
    return auto_smooth, auto_brightness

def apply_beauty_filter(image, smooth_level=1.0, brightness=1.0, shape_factor=0.1, acne_intensity=1.0, auto_adjust=False):
    if auto_adjust:
        auto_smooth, auto_bright = auto_adjust_parameters(image)
        smooth_level = min(auto_smooth, 2.0)
        brightness = auto_bright
    
    #shaped = apply_face_shape(image, shape_factor)
    shaped = image
    pil_img = Image.fromarray(shaped)
    enhanced = skin_enhancer.enhance(pil_img)
    
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.multiply(v, brightness)
    hsv = cv2.merge([h, s, v])
    final = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return final

# Streamlit UI
st.title("Beauty Filter App")

# Parameters
auto_adjust = st.checkbox("Auto-Adjust Smoothing & Brightness", value=False)
smooth_level = st.slider("Smoothing Level", 0.1, 2.0, 1.0, disabled=auto_adjust)
brightness = st.slider("Brightness", 0.5, 1.5, 1.0, disabled=auto_adjust)
shape_factor = st.slider("Face Shape", 0.0, 0.5, 0.1)
acne_intensity = st.slider("Acne Removal Intensity", 0.5, 2.0, 1.0)

# File Upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    processed_image = apply_beauty_filter(image, smooth_level, brightness, shape_factor, acne_intensity, auto_adjust)
    combined = np.hstack((image, processed_image))
    st.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), caption="Original (Left) | Processed (Right)")

# Run with: streamlit run app.py