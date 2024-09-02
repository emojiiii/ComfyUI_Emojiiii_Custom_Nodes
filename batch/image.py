import argparse
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
    

def detect_faces(image):
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return faces

def crop_image(image_path, output_path, width=None, height=None, crop_face=False, format=None, quality=95):
    # Load the original image using Pillow
    image_path = Path(image_path)
    img = Image.open(image_path)

    # Load the original image using OpenCV
    img_cv = cv2.imread(str(image_path))
    
    # Detect faces if required
    if crop_face:
        faces = detect_faces(img_cv)
        
        if len(faces) == 0:
            print(f"No faces detected: {image_path}")
            center_x, center_y = img.width // 2, img.height // 2
            return
        
        # Assume the first face is the one we want to crop around
        x, y, w, h = faces[0]
        
        # Calculate the center of the face
        center_x, center_y = x + w // 2, y + h // 2
    else:
        # If not cropping by face, use the center of the image
        center_x, center_y = img.width // 2, img.height // 2
    
    # Determine the new dimensions
    if width and height:
        new_width, new_height = width, height
    elif width:
        scale_factor = width / img.width
        new_height = int(img.height * scale_factor)
        new_width = width
    elif height:
        scale_factor = height / img.height
        new_width = int(img.width * scale_factor)
        new_height = height
    else:
        raise ValueError("Either width or height must be provided.")
    
    # Calculate the new boundaries
    left = center_x - new_width // 2
    top = center_y - new_height // 2
    right = center_x + new_width // 2
    bottom = center_y + new_height // 2
    
    # Adjust boundaries if they exceed the image dimensions
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > img.width:
        left -= right - img.width
        right = img.width
    if bottom > img.height:
        top -= bottom - img.height
        bottom = img.height
    
    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))

    # only support format to be jpg or png or webp or jpeg
    if format not in ["jpg", "jpeg", "png"]:
        format = "png"
    
    # Save the cropped image
    cropped_img.save(output_path, format=format, quality=quality)
