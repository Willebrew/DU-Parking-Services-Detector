"""
This script downsamples an image to a specific size using the Python Imaging Library.
"""
from PIL import Image


def downsample_image(image_path, output_path):
    """
    Downsample the input image to 256x256 pixels and save it to the output path.
    :param image_path:
    :param output_path:
    :return:
    """
    image = Image.open(image_path)

    resized_image = image.resize((256, 256))

    resized_image.save(output_path)

    print(f"Image downsampled and saved as {output_path}")


INPUT_IMAGE_PATH = "Dataset/images/246bee58-images-2.jpg"
OUTPUT_IMAGE_PATH = "output-image.png"

downsample_image(INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH)
