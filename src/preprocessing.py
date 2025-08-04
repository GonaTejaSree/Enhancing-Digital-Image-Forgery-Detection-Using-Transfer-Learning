import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import os

def perform_ela(image_path, resave_quality=90):
    """
    Perform Error Level Analysis (ELA) on an image.
    :param image_path: path to original image
    :param resave_quality: quality to resave the image (default = 90)
    :return: ELA image as NumPy array
    """
    temp_filename = "temp_ela.jpg"
    original = Image.open(image_path).convert("RGB")
    
    # Save and reload at lower quality
    original.save(temp_filename, "JPEG", quality=resave_quality)
    compressed = Image.open(temp_filename)

    # Get pixel differences
    ela_image = ImageChops.difference(original, compressed)

    # Enhance differences
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return np.array(ela_image)

def resize_image(img_array, size=(224, 224)):
    """
    Resize image to target size (for CNN input).
    :param img_array: image as NumPy array
    :param size: (width, height) tuple
    :return: resized image as NumPy array
    """
    return cv2.resize(img_array, size)

def preprocess_image(image_path, size=(224, 224)):
    """
    Full preprocessing pipeline: ELA + Resize
    :param image_path: path to input image
    :return: preprocessed image array
    """
    ela_img = perform_ela(image_path)
    ela_img = resize_image(ela_img, size)
    ela_img = ela_img / 255.0  # Normalize to [0, 1]
    return ela_img

def load_dataset_from_folder(folder_path, size=(224, 224)):
    """
    Load and preprocess all images in a folder.
    :param folder_path: path to folder
    :return: tuple of (X, filenames)
    """
    X = []
    filenames = []

    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, file)
            try:
                img = preprocess_image(img_path, size=size)
                X.append(img)
                filenames.append(file)
            except Exception as e:
                print(f"Skipping {file}: {e}")

    return np.array(X), filenames
