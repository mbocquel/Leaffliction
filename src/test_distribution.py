import unittest
from Distribution import Distribution

class TestDistribution(unittest.TestCase):
    """Test the distribution class"""
    
    def test_1(self):
        distribution = Distribution("../img_for_test/Distribution/test1")
        file_count = distribution.getFileCount()
        self.assertIsInstance(file_count, dict)
        target = {
            'Apple_Black_rot' : 3,
            'Apple_healthy': 4,
            'Grape_Esca' : 8
        }
        self.assertDictEqual(file_count, target)

    def test1_list(self):
        distribution = Distribution("../img_for_test/Distribution/test1")
        file_list = distribution.getFileList()
        self.assertIsInstance(file_list, dict)
        target = {
            'Apple_Black_rot' : set([
                "../img_for_test/Distribution/test1/Apple_Black_rot/image (1).JPG",
                "../img_for_test/Distribution/test1/Apple_Black_rot/image (2).JPG",
                "../img_for_test/Distribution/test1/Apple_Black_rot/image (4).JPG"]),
            'Apple_healthy': set([
                "../img_for_test/Distribution/test1/Apple_healthy/image (1).JPG",
                "../img_for_test/Distribution/test1/Apple_healthy/sub/image (4).JPG",
                "../img_for_test/Distribution/test1/Apple_healthy/image (2).JPG", 
                "../img_for_test/Distribution/test1/Apple_healthy/image (3).JPG"]),
            'Grape_Esca' : set([
                "../img_for_test/Distribution/test1/Grape_Esca/image (12).JPG",
                "../img_for_test/Distribution/test1/Grape_Esca/image (13).JPG",
                "../img_for_test/Distribution/test1/Grape_Esca/image (14).JPG",
                "../img_for_test/Distribution/test1/Grape_Esca/image (15).JPG",
                "../img_for_test/Distribution/test1/Grape_Esca/image (16).JPG",
                "../img_for_test/Distribution/test1/Grape_Esca/image (17).JPG",
                "../img_for_test/Distribution/test1/Grape_Esca/image (18).JPG",
                "../img_for_test/Distribution/test1/Grape_Esca/image (19).JPG",])
        }
        self.assertDictEqual(file_list, target)
    
    def test_2_empty(self):
        distribution = Distribution("../img_for_test/Distribution/test2")
        file_count = distribution.getFileCount()
        self.assertIsInstance(file_count, dict)
        target = {}
        self.assertDictEqual(file_count, target)

    def test_3_empty_cat(self):
        distribution = Distribution("../img_for_test/Distribution/test3")
        data = distribution.getFileCount()
        self.assertIsInstance(data, dict)
        target = {
            'Apple_Black_rot': 2,
            'Apple_healthy': 0
        }
        self.assertDictEqual(data, target)

if __name__ == "__main__":
    unittest.main()