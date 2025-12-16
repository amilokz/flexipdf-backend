from pdf2docx import Converter
from pdf2image import convert_from_path
from PIL import Image
import os
import subprocess

def pdf_to_word(input_path, output_path):
    """
    Convert PDF to Word document using pdf2docx
    """
    try:
        cv = Converter(input_path)
        cv.convert(output_path, start=0, end=None)
        cv.close()
        print(f"[SUCCESS] PDF converted to Word: {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] PDF to Word conversion failed: {e}")
        raise e




def pdf_to_images(input_path, output_folder, dpi=200):
    """
    Convert each PDF page to separate images
    Returns list of image file paths
    """
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Convert PDF to images
        images = convert_from_path(input_path, dpi=dpi)
        
        image_paths = []
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        for i, image in enumerate(images):
            # Save each page as separate image
            output_filename = f"{base_name}_page_{i+1}.png"
            output_path = os.path.join(output_folder, output_filename)
            image.save(output_path, 'PNG')
            image_paths.append(output_path)
            print(f"[SUCCESS] Page {i+1} saved as: {output_filename}")
        
        return image_paths
    except Exception as e:
        print(f"[ERROR] PDF to Images conversion failed: {e}")
        raise e

def images_to_pdf(image_paths, output_path):
    """
    Merge multiple images into a single PDF
    """
    try:
        # Open first image to get size
        first_image = Image.open(image_paths[0])
        
        # Create list of images
        images = [Image.open(img).convert('RGB') for img in image_paths]
        
        # Save as PDF
        images[0].save(output_path, save_all=True, append_images=images[1:])
        
        print(f"[SUCCESS] {len(image_paths)} images merged into PDF: {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Images to PDF conversion failed: {e}")
        raise e