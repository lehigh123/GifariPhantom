import logging
from PIL import Image, ImageOps
import base64
from io import BytesIO


def decode_image_entry(image_entry, is_base64=False):
    """
    Decode an image entry from a file path or a base64 string into a PIL Image.
    Args:
        image_entry (str): File path or base64 string.
        is_base64 (bool): Whether the entry is a base64 string.
    Returns:
        PIL.Image.Image: Decoded image.
    """
    if is_base64:
        image_data = base64.b64decode(image_entry)
        img = Image.open(BytesIO(image_data)).convert("RGB")
    else:
        with Image.open(image_entry) as img_file:
            img = img_file.convert("RGB")
    return img


def resize_and_pad_image(img, target_size):
    """
    Resize and pad the image to the target size while preserving aspect ratio.
    Args:
        img (PIL.Image.Image): Input image.
        target_size (tuple): (width, height)
    Returns:
        PIL.Image.Image: Processed image.
    """
    w, h = target_size
    img_ratio = img.width / img.height
    target_ratio = w / h

    if img_ratio > target_ratio:  # Image is wider than target
        new_width = w
        new_height = int(new_width / img_ratio)
    else:  # Image is taller than target
        new_height = h
        new_width = int(new_height * img_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    delta_w = w - img.size[0]
    delta_h = h - img.size[1]
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )
    new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))
    return new_img


def load_ref_images(paths, size, is_base64=False):
    """
    Load and process reference images to match the target size while maintaining aspect ratio.
    Args:
        paths (str): Comma-separated string of image paths or base64 strings
        size (tuple): Target size as (width, height)
        is_base64 (bool): If True, treat each entry as a base64-encoded image string
    Returns:
        list: List of processed PIL Image objects
    """
    if not paths or paths.strip() == "":
        raise ValueError("Image path cannot be empty or null")

    w, h = size[0], size[1]
    ref_entries = paths.split(",")
    ref_images = []
    for entry in ref_entries:
        img = decode_image_entry(entry, is_base64)
        processed_img = resize_and_pad_image(img, (w, h))
        ref_images.append(processed_img)
    logging.info(f"ref_images loaded {ref_images}")
    return ref_images

def image_path_to_base64(image_path):
    """
    Convert an image at the given path to a base64-encoded string.
    Args:
        image_path (str): Path to the image file.
    Returns:
        str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string 

def base64_to_image(base64_str):
    """
    Decode a base64-encoded image string and return a PIL Image object.
    Args:
        base64_str (str): Base64-encoded string of the image.
    Returns:
        PIL.Image.Image: Decoded image object.
    """
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)) 