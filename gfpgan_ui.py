import streamlit as st
import cv2
import numpy as np
import torch
from basicsr.utils import imwrite
from gfpgan import GFPGANer
import os
from PIL import Image

# Function to set up GFPGAN
def setup_gfpgan(version='1.3', upscale=2, bg_upsampler='realesrgan'):
    if version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    else:
        raise ValueError(f'Unsupported model version {version}')

    model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = url

    bg_upsampler_obj = None
    if bg_upsampler == 'realesrgan' and torch.cuda.is_available():
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler_obj = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True
        )

    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler_obj
    )
    return restorer

# Process image function
def process_image(restorer, img, weight=0.5):
    if isinstance(img, Image.Image):
        img = np.array(img)

    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    _, _, restored_img = restorer.enhance(
        img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=weight
    )

    if restored_img is not None:
        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)

    return restored_img

# Streamlit UI
def main():
    st.title("GFPGAN Face Restoration")
    st.write("Upload an image to enhance faces using GFPGAN")

    # Sidebar options
    st.sidebar.header("Settings")
    version = st.sidebar.selectbox("Model Version", ["1", "1.3"], index=1)
    upscale = st.sidebar.slider("Upscale Factor", 1, 4, 2)
    weight = st.sidebar.slider("Restoration Strength", 0.0, 1.0, 0.5)
    use_bg_upsampler = st.sidebar.checkbox("Use Background Upsampler", value=True)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)

        # Load original image
        original_image = Image.open(uploaded_file)

        # Display original image in first column
        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_column_width=True)

        # Process image automatically
        with st.spinner("Enhancing image..."):
            # Setup GFPGAN
            bg_upsampler = 'realesrgan' if use_bg_upsampler else None
            restorer = setup_gfpgan(version=version, upscale=upscale, bg_upsampler=bg_upsampler)

            # Process image
            restored_image = process_image(restorer, original_image, weight=weight)

            # Display restored image in second column
            with col2:
                st.subheader("Enhanced Image")
                st.image(restored_image, use_column_width=True)

            # Download button below the images
            restored_pil = Image.fromarray(restored_image)
            restored_pil.save("restored_image.png")
            with open("restored_image.png", "rb") as file:
                st.download_button(
                    label="Download Restored Image",
                    data=file,
                    file_name="restored_image.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()