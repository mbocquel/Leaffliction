import unittest
from Augmentation import Augmentation
from PIL import Image
import os
import numpy as np


class TestAugmentation(unittest.TestCase):
    """Test the Augmentation class"""
    def setUp(self):
        """Set up the test"""
        self.img = np.array(Image.open("../img_for_test/Augmentation/img.JPG"))
        self.aug = Augmentation("../img_for_test/Augmentation/img.JPG")
        
    def test_1_rotating(self):
        """Test the rotation"""
        expected_rot = np.array(Image.open("../img_for_test/Augmentation/expected/rotated.png"))
        rotated_30 = np.array(self.aug.rotate_img(save=False))
        rotated_60 = np.array(self.aug.rotate_img(rot_angle=60, save=False))
        self.assertTrue(np.array_equal(rotated_30, expected_rot))
        self.assertFalse(np.array_equal(rotated_60, expected_rot))
        self.assertFalse(np.array_equal(rotated_30, self.img))
    
    def test_2_flip(self):
        """Test the Flip"""
        expected_fliped = np.array(Image.open("../img_for_test/Augmentation/expected/fliped.png"))
        flip = np.array(self.aug.flip_img(save=False))
        self.assertTrue(np.array_equal(flip, expected_fliped))
        self.assertFalse(np.array_equal(flip, self.img))
    
    def test_3_blur(self):
        """Test the blur"""
        expected_blur = np.array(Image.open("../img_for_test/Augmentation/expected/blured.png"))
        blur = np.array(self.aug.blur_img(save=False))
        self.assertTrue(np.array_equal(blur, expected_blur))
        self.assertFalse(np.array_equal(blur, self.img))

    def test_4_illum(self):
        """Test the illumination"""
        expected_illum = np.array(Image.open("../img_for_test/Augmentation/expected/illuminated.png"))
        illum = np.array(self.aug.illuminate_img(illum_level=1.5, save=False))
        self.assertTrue(np.array_equal(illum, expected_illum))
        self.assertFalse(np.array_equal(illum, self.img))
        self.assertTrue(np.array_equal(self.aug.illuminate_img(illum_level=1), self.img))
    
    def test_5_scal(self):
        """Test the scaling"""
        expected_scal = np.array(Image.open("../img_for_test/Augmentation/expected/scaled.png"))
        scal_5 = np.array(self.aug.scale_img(zoom=5, save=False))
        scal_10 = np.array(self.aug.scale_img(zoom=10, save=False))
        scal_1 = np.array(self.aug.scale_img(zoom=1, save=False))
        self.assertTrue(np.array_equal(scal_5, expected_scal))
        self.assertFalse(np.array_equal(scal_5, self.img))
        self.assertFalse(np.array_equal(scal_10, expected_scal))
        self.assertTrue(np.array_equal(scal_1, self.img))
    
    def test_5_contr(self):
        """Test the contrast"""
        expected_contr = np.array(Image.open("../img_for_test/Augmentation/expected/contrasted.png"))
        contr = np.array(self.aug.increase_contrast(cont_factor=1.5, save=False))
        self.assertTrue(np.array_equal(contr, expected_contr))
        self.assertFalse(np.array_equal(contr, self.img))
        self.assertTrue(np.array_equal(self.aug.increase_contrast(cont_factor=1), self.img))

    def test_6_saving_from_each_generation(self):
        """Test the saving option of individual generation"""
        self.aug.rotate_img()
        self.assertTrue(os.path.isfile("../img_for_test/Augmentation/img_rotated_30.png"))
        os.remove("../img_for_test/Augmentation/img_rotated_30.png")
        self.aug.flip_img()
        self.assertTrue(os.path.isfile("../img_for_test/Augmentation/img_fliped.png"))
        os.remove("../img_for_test/Augmentation/img_fliped.png")
        self.aug.blur_img()
        self.assertTrue(os.path.isfile("../img_for_test/Augmentation/img_blured.png"))
        os.remove("../img_for_test/Augmentation/img_blured.png")
        self.aug.illuminate_img()
        self.assertTrue(os.path.isfile("../img_for_test/Augmentation/img_illuminated_1.5.png"))
        os.remove("../img_for_test/Augmentation/img_illuminated_1.5.png")
        self.aug.scale_img()
        self.assertTrue(os.path.isfile("../img_for_test/Augmentation/img_scaled_5.png"))
        os.remove("../img_for_test/Augmentation/img_scaled_5.png")
        self.aug.increase_contrast()
        self.assertTrue(os.path.isfile("../img_for_test/Augmentation/img_contrasted_1.5.png"))
        os.remove("../img_for_test/Augmentation/img_contrasted_1.5.png")

    def test_7_saving(self):
        """Test the global saving option"""
        self.aug.generate_augmented_imgs()
        self.assertTrue(os.path.isfile("../img_for_test/Augmentation/img_rotated_30.png"))
        self.assertTrue(os.path.isfile("../img_for_test/Augmentation/img_fliped.png"))
        self.assertTrue(os.path.isfile("../img_for_test/Augmentation/img_blured.png"))
        self.assertTrue(os.path.isfile("../img_for_test/Augmentation/img_illuminated_1.5.png"))
        self.assertTrue(os.path.isfile("../img_for_test/Augmentation/img_scaled_5.png"))
        self.assertTrue(os.path.isfile("../img_for_test/Augmentation/img_contrasted_1.5.png"))
        os.remove("../img_for_test/Augmentation/img_rotated_30.png")
        os.remove("../img_for_test/Augmentation/img_fliped.png")
        os.remove("../img_for_test/Augmentation/img_blured.png")
        os.remove("../img_for_test/Augmentation/img_illuminated_1.5.png")
        os.remove("../img_for_test/Augmentation/img_scaled_5.png")
        os.remove("../img_for_test/Augmentation/img_contrasted_1.5.png")


if __name__ == "__main__":
    unittest.main()