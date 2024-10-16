"""
convert_heic.py

This script converts HEIC images to PNG format. It reads all HEIC files from the
specified input directory, converts them to PNG using the Pillow library, and
saves the converted images to the specified output directory.

Functions:
    None

Dependencies:
    - os
    - PIL (Pillow)
    - pillow_heif

Usage:
    Ensure the input directory contains HEIC files and the output directory exists or
    will be created. Run the script to convert all HEIC files in the input directory to PNG format.
"""
import os
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

INPUT_DIRECTORY = 'DatasetHEIC'
OUTPUT_DIRECTORY = 'PNG'

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

for filename in os.listdir(INPUT_DIRECTORY):
    if filename.lower().endswith('.heic'):
        input_path = os.path.join(INPUT_DIRECTORY, filename)
        output_path = os.path.join(OUTPUT_DIRECTORY, os.path.splitext(filename)[0] + '.png')

        try:
            with Image.open(input_path) as img:
                img.save(output_path, 'PNG')
            print(f"Converted {filename} to PNG")
        except (IOError, OSError) as e:
            print(f"Error converting {filename}: {str(e)}")

print("!Conversion complete!")
