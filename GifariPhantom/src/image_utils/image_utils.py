import logging
from PIL import Image, ImageOps

def load_ref_images(path, size):
    """
    Load and process reference images to match the target size while maintaining aspect ratio.
    
    Args:
        path (str): Comma-separated string of image paths
        size (tuple): Target size as (width, height)
    
    Returns:
        list: List of processed PIL Image objects
    """
    if not path or path.strip() == "":
        raise ValueError("Image path cannot be empty or null")
        
    # Load size.
    h, w = size[1], size[0]
    # Load images.
    ref_paths = path.split(",")
    ref_images = []
    for image_path in ref_paths:
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            # Calculate the required size to keep aspect ratio and fill the rest with padding.
            img_ratio = img.width / img.height
            target_ratio = w / h

            if img_ratio > target_ratio:  # Image is wider than target
                new_width = w
                new_height = int(new_width / img_ratio)
            else:  # Image is taller than target
                new_height = h
                new_width = int(new_height * img_ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Create a new image with the target size and place the resized image in the center
            delta_w = w - img.size[0]
            delta_h = h - img.size[1]
            padding = (
                delta_w // 2,
                delta_h // 2,
                delta_w - (delta_w // 2),
                delta_h - (delta_h // 2),
            )
            new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))
            ref_images.append(new_img)
    logging.info(f"ref_images loaded {ref_images}")
    return ref_images 