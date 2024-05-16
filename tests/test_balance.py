import unittest
from src.Balance import Balance
from src.Distribution import Distribution
import os

class TestBalance(unittest.TestCase):
    """Test the distribution class"""
    
    def setUp(self):
        """Set up for the tests."""
        self.test_dirs = ["img_for_test/Balance/test1", 
                          "img_for_test/Balance/test2", 
                          "img_for_test/Balance/test3", 
                          "img_for_test/Balance/test4"]
        for test_dir in self.test_dirs:
            assert os.path.isdir(test_dir), f"Test directory {test_dir} does not exist."


    def tearDown(self):
        """Clean up after each test."""
        for test_dir in self.test_dirs:
            balance = Balance(test_dir)
            balance.remove_augmented_img()


    def test_1(self):
        balance = Balance(self.test_dirs[0])
        distribution = Distribution(self.test_dirs[0])
        self.assertTrue(balance.balance_status == "Not Balanced")
        self.assertFalse(balance.isBalanced)
        balance.balance_data()
        self.assertTrue(balance.balance_status == "Balanced")
        self.assertTrue(balance.isBalanced)
        balanced_distribution = distribution.getFileCount(reload=True)
        target = {
            'Apple_Black_rot': 8,
            'Apple_healthy': 8,
            'Grape_Esca' : 8
        }
        self.assertDictEqual(balanced_distribution, target)

    
    def test_2_empty(self):
        balance = Balance(self.test_dirs[1])
        distribution = Distribution(self.test_dirs[1])
        self.assertFalse(balance.isBalanced)
        self.assertTrue(balance.balance_status == "Empty dataset")
        balance.balance_data()
        self.assertFalse(balance.isBalanced)
        self.assertTrue(balance.balance_status == "Empty dataset")
        self.assertDictEqual(distribution.getFileCount(reload=True), {})


    def test_3_empty_cat(self):
        balance = Balance(self.test_dirs[2])
        distribution = Distribution(self.test_dirs[2])
        self.assertFalse(balance.isBalanced)
        balance.balance_data()
        self.assertTrue(balance.balance_status == "Empty category, imposible to balance")
        target = {
            'Apple_Black_rot': 2,
            'Apple_healthy': 0
        }
        self.assertFalse(balance.isBalanced)
        self.assertDictEqual(distribution.getFileCount(reload=True), target)


    def test_4_already_balanced(self):
        balance = Balance(self.test_dirs[3])
        distribution = Distribution(self.test_dirs[3])
        init_distrib = distribution.getFileCount()
        self.assertTrue(balance.isBalanced)
        self.assertTrue(balance.balance_status == "Balanced")
        balance.balance_data()
        self.assertTrue(balance.balance_status == "Balanced")
        self.assertTrue(balance.isBalanced)
        final_distrib = distribution.getFileCount(reload=True)
        self.assertDictEqual(init_distrib, final_distrib)


if __name__ == "__main__":
    unittest.main()