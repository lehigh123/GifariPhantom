import unittest
from PIL import Image
import os
import tempfile
from src.image_utils.image_utils import load_ref_images

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

    # def test_invalid_size(self):
    #     """Test handling of invalid size"""
    #     with self.assertRaises(ValueError):
    #         load_ref_images(self.square_path, (0, 200))
    #     with self.assertRaises(ValueError):
    #         load_ref_images(self.square_path, (200, 0))

if __name__ == '__main__':
    unittest.main() 