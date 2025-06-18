import unittest
from PIL import Image
import os
import tempfile
from src.image_utils.image_utils import load_ref_images, image_path_to_base64

class TestLoadRefImages(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test images
        self.test_dir = tempfile.mkdtemp()
        
        # Create test images with different aspect ratios
        # Square image
        self.square_img = Image.new('RGB', (100, 100), color='red')
        self.square_path = os.path.join(self.test_dir, 'square.png')
        self.square_img.save(self.square_path)
        
        # Wide image
        self.wide_img = Image.new('RGB', (200, 100), color='blue')
        self.wide_path = os.path.join(self.test_dir, 'wide.png')
        self.wide_img.save(self.wide_path)
        
        # Tall image
        self.tall_img = Image.new('RGB', (100, 200), color='green')
        self.tall_path = os.path.join(self.test_dir, 'tall.png')
        self.tall_img.save(self.tall_path)
        # Ref image paths 
        self.test_image_1_path = 'tests/test-data/ref1.png'
        self.test_image_2_path = 'tests/test-data/ref1.png'

    def tearDown(self):
        # Clean up temporary files
        for file in [self.square_path, self.wide_path, self.tall_path]:
            if os.path.exists(file):
                os.remove(file)
        os.rmdir(self.test_dir)

    def test_single_image(self):
        """Test loading a single image"""
        result = load_ref_images(self.square_path, (200, 200))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].size, (200, 200))

    def test_multiple_images(self):
        """Test loading multiple images"""
        paths = f"{self.square_path},{self.wide_path},{self.tall_path}"
        result = load_ref_images(paths, (200, 200))
        self.assertEqual(len(result), 3)
        for img in result:
            self.assertEqual(img.size, (200, 200))

    def test_aspect_ratio_preservation(self):
        """Test that aspect ratio is preserved with padding"""
        # Test wide image
        result = load_ref_images(self.wide_path, (200, 200))
        self.assertEqual(result[0].size, (200, 200))
        # Check that the image is centered (pixels at edges should be white)
        self.assertEqual(result[0].getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(result[0].getpixel((199, 199)), (255, 255, 255))

        # Test tall image
        result = load_ref_images(self.tall_path, (200, 200))
        self.assertEqual(result[0].size, (200, 200))
        # Check that the image is centered
        self.assertEqual(result[0].getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(result[0].getpixel((199, 199)), (255, 255, 255))

    def test_empty_path(self):
        """Test handling of empty path"""
        with self.assertRaises(ValueError):
            load_ref_images("", (200, 200))

    def test_invalid_path(self):
        """Test handling of invalid path"""
        with self.assertRaises(FileNotFoundError):
            load_ref_images("nonexistent.png", (200, 200))

    def test_load_single_reference_image(self):
        result = load_ref_images(self.test_image_1_path, (1280, 720))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].size, (1280, 720))

    def test_load_multiple_reference_images(self):
        result = load_ref_images(f"{self.test_image_1_path},{self.test_image_2_path}", (1280, 720))
        self.assertEqual(len(result), 2)
        for img in result:
            self.assertEqual(img.size, (1280, 720))

    def test_image_path_to_base64(self):
        # Use the square image created in setUp
        base64_str = image_path_to_base64(self.square_path)
        self.assertIsInstance(base64_str, str)
        self.assertTrue(len(base64_str) > 0)
        # Check that decoding the base64 string gives the original image bytes
        with open(self.square_path, "rb") as img_file:
            original_bytes = img_file.read()
        import base64
        decoded_bytes = base64.b64decode(base64_str)
        self.assertEqual(original_bytes, decoded_bytes)

    def test_base64_encode_decode_image(self):
        # Encode the image to base64
        base64_str = image_path_to_base64(self.square_path)
        # Decode the image from base64
        from src.image_utils.image_utils import base64_to_image
        decoded_img = base64_to_image(base64_str)
        # Open the original image
        with Image.open(self.square_path) as original_img:
            original_img = original_img.convert("RGB")
            # Compare size
            self.assertEqual(decoded_img.size, original_img.size)
            # Compare pixel data
            self.assertEqual(list(decoded_img.getdata()), list(original_img.getdata()))

    def test_load_ref_images_with_base64(self):
        base64_str = image_path_to_base64(self.test_image_1_path)
        base_64_loaded_image = load_ref_images(base64_str, (1280, 720), is_base64=True)
        local_image = load_ref_images(self.test_image_1_path, (1280, 720), False)
        # Encode the image to base64
        self.assertEqual(base_64_loaded_image[0].size, local_image[0].size)
        self.assertEqual(list(base_64_loaded_image[0].getdata()), list(local_image[0].getdata()))

    def test_load_multiple_ref_images_with_base64(self):
        base64_str_1 = image_path_to_base64(self.test_image_1_path)
        base64_str_2 = image_path_to_base64(self.test_image_2_path)
        base64_concat = f"{base64_str_1},{base64_str_2}"
        base64_loaded_images = load_ref_images(base64_concat, (1280, 720), is_base64=True)
        local_loaded_images = load_ref_images(f"{self.test_image_1_path},{self.test_image_2_path}", (1280, 720), False)
        self.assertEqual(len(base64_loaded_images), 2)
        self.assertEqual(len(local_loaded_images), 2)
        for img_b64, img_local in zip(base64_loaded_images, local_loaded_images):
            self.assertEqual(img_b64.size, img_local.size)
            self.assertEqual(list(img_b64.getdata()), list(img_local.getdata()))

if __name__ == '__main__':
    unittest.main() 