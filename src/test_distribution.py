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
    
    def test_2_empty(self):
        distribution = Distribution("../img_for_test/Distribution/test2")
        file_count = distribution.getFileCount()
        self.assertIsInstance(file_count, dict)
        target = {}
        self.assertDictEqual(file_count, target)

    def test_3_empty_cat(self):
        distributionTest1 = Distribution("../img_for_test/Distribution/test3")
        data = distributionTest1.getFileCount()
        self.assertIsInstance(data, dict)
        target = {
            'Apple_Black_rot': 2,
            'Apple_healthy': 0
        }
        self.assertDictEqual(data, target)

if __name__ == "__main__":
    unittest.main()