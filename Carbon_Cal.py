import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import easyocr
import requests
import io
import uuid
import cv2
from PIL import Image
import torch
from ultralytics import YOLO

# Initialize EasyOCR reader
reader = easyocr.Reader(["en"])

# Load YOLO model
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model

st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")

selected = option_menu(
    menu_title="Carbon Closet Calculator",
    options=["Product Details", "Production Phase Calculation",
             "Manufacturing Phase Calculation",
             "Transportation Phase Calculation", "User Phase Calculation",
             "Overall Dashboard"],
    icons=["bi bi-cart3", "bi bi-tree-fill", "bi bi-buildings-fill", "bi bi-fuel-pump", "bi bi-droplet-half",
           "bi bi-graph-down"],
    menu_icon="bi bi-calculator-fill",
    default_index=0,
    orientation="horizontal"
)

if selected == "Product Details":
    st.title("Product Details")
    
    # Create two containers
    left_container, right_container = st.columns([2, 3])
    
    with left_container:
        st.subheader("Clothes Details")
        
        st.subheader("Capture or Upload Clothes Image")
        captured_image = st.camera_input("Take a Picture")
        uploaded_clothes_images = st.file_uploader("Upload up to 6 images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="clothes_images")
        
        st.subheader("Tag Images")
        uploaded_tag_images = st.file_uploader("Upload a tag image...", type=["jpg", "png", "jpeg"], key="tag_images")
        
        st.subheader("Category")
        category = st.text_input("Category")
        
        st.subheader("Skinny Jeans")
        skinny_jeans = st.selectbox("Select the brand", ["Brand 1", "Brand 2", "Brand 3"])
        
        st.subheader("Size")
        size = st.selectbox("Select the size", ["Small", "Medium", "Large"])
        st.write("Choose the size that matches the item label. See the size guide")
        
        st.subheader("State")
        state = st.selectbox("Indicate the condition of your item", ["New", "Used - Like New", "Used - Good"])
        
        st.subheader("Color")
        color = st.multiselect("Choose 2 colors maximum", ["Red", "Blue", "Green", "Yellow", "Black"])
        
        st.subheader("Material (recommended)")
        material = st.text_area("Material")
    
    with right_container:
        st.subheader("Clothes Images")
        images = []
        if captured_image:
            images.append(captured_image)
        if uploaded_clothes_images:
            images.extend(uploaded_clothes_images)
        
        for image_bytes in images:
            try:
                image = Image.open(io.BytesIO(image_bytes.read()))
                st.image(image, caption="Clothes Image", use_container_width=True)
                
                # Convert image to numpy array for OCR
                img_np = np.array(image.convert('RGB'))
                ocr_result = reader.readtext(img_np)
                extracted_text = ' '.join([res[1] for res in ocr_result])
                
                st.text_input("Extracted Text", extracted_text)
                
                # Run YOLO object detection
                results = model(img_np)
                detected_objects = [model.names[int(pred[5])] for pred in results[0].boxes.data]
                st.write("Detected Objects:", detected_objects)
                
            except Exception as e:
                st.warning(f"Error processing Clothes Image: {e}")
        
        st.subheader("Tag Images")
        if uploaded_tag_images:
            for image_bytes in uploaded_tag_images:
                try:
                    image = Image.open(io.BytesIO(image_bytes.read()))
                    st.image(image, caption="Tag Image", use_container_width=True)
                except Exception as e:
                    st.warning(f"Error processing Tag Image: {e}")

if selected == "Production Phase Calculation":
    st.title("Carbon emission calculation at production phase")

if selected == "Manufacturing Phase Calculation":
    st.title("Carbon emission calculation at manufacturing phase")

if selected == "Transportation Phase Calculation":
    st.title("Carbon emission calculation at transportation phase")

if selected == "User Phase Calculation":
    st.title("Carbon emission calculation at user phase")

if selected == "Overall Dashboard":
    st.title("Overall scenario for the carbon emission")
