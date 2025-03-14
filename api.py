from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
from io import BytesIO
from PIL import Image
from insightface.app import FaceAnalysis
import torch
from torchvision import transforms
import cv2  # Minimal use for I/O only

app = FastAPI()

# Initialize InsightFace
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Simulated AI model for skin smoothing and acne removal
class SkinEnhancer:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.model = lambda x: x  # Placeholder for real model

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
    
    # Define control points for warping
    src_pts = landmarks.astype(np.float32)
    dst_pts = src_pts.copy()
    
    # Slim the face by adjusting jawline points toward the center
    center_x = np.mean(src_pts[:, 0])
    for i in range(len(src_pts)):
        x, y = src_pts[i]
        dx = center_x - x
        dist = abs(dx)
        if dist > 0 and i < 34:  # Focus on jawline (first 34 points)
            shift = shape_factor * dist
            new_x = x + (dx * shift / dist)
            dst_pts[i] = [new_x, y]
    
    # Use affine transform for simplicity and alignment
    M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])  # Use first 3 points for stability
    warped = cv2.warpAffine(image, M, (w, h))
    
    return warped

def apply_beauty_filter(image, smooth_level=1.0, brightness=1.0, shape_factor=0.1, acne_intensity=1.0):
    # Apply face shape adjustment
    shaped = apply_face_shape(image, shape_factor)
    
    # Simulate AI skin enhancement
    pil_img = Image.fromarray(shaped)
    enhanced = skin_enhancer.enhance(pil_img)
    
    # Adjust brightness (minimal OpenCV)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.multiply(v, brightness)
    hsv = cv2.merge([h, s, v])
    final = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return final

@app.post("/beautify")
async def beautify_image(file: UploadFile = File(...), smooth_level: float = 1.0, brightness: float = 1.0, 
                        shape_factor: float = 0.1, acne_intensity: float = 1.0):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if original is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    processed = apply_beauty_filter(original, smooth_level, brightness, shape_factor, acne_intensity)
    combined = np.hstack((original, processed))
    
    _, buffer = cv2.imencode(".jpg", combined)
    img_io = BytesIO(buffer)
    
    return StreamingResponse(img_io, media_type="image/jpeg")

# Run with: uvicorn api:app --reload