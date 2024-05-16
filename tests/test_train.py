import unittest
from src.Train import MyLeafCNN
from src.Train import normalise_img, create_zip
import numpy as np


class TestTrainMyLeafCNN(unittest.TestCase):
    """Test for the trainning of my CNN Model for Leaf identifiaction"""

    def setUp(self) -> None:
        
        return super().setUp()


    def tearDown(self) -> None:
        return super().tearDown()


    def test_normalise_img(self):
        """ Test the normilise img function"""
        img = np.random.randint(255, size=(256, 256, 3), dtype=np.uint8)
        norm_img = normalise_img(img)
        self.assertTrue(np.max(norm_img.numpy()) <= 1.0)


    def test_load_dataset(self) -> int:
        """ Test the load_dataset function"""
        return ""


    def test_create_zip_file(self):
        """Test the create zip file function"""


    def test_build_model(self):
        """Test the build model function"""

    
    def test_partition_dataset(self):
        """Test the partition dataset function"""


    def test_train_model(self):
        """Test the train function"""


if __name__ == "__main__":
    unittest.main()