import unittest
from Balance import Balance
from Distribution import Distribution

class TestBalance(unittest.TestCase):
    """Test the distribution class"""
    
    def test_1(self):
        balance = Balance("../img_for_test/Distribution/test1")
        distribution = Distribution("../img_for_test/Distribution/test1")
        self.assertTrue(balance.balance_statut == "Not Balanced")
        self.assertFalse(balance.isBalanced)
        balance.balance_data()
        self.assertTrue(balance.balance_statut == "Balanced")
        self.assertTrue(balance.isBalanced)
        balanced_distribution = distribution.getFileCount()
        target = {
            'Apple_Black_rot': 8,
            'Apple_healthy': 8,
            'Grape_Esca' : 8
        }
        self.assertDictEqual(balanced_distribution, target)
    
    def test_2_empty(self):
        balance = Balance("../img_for_test/Distribution/test2")
        distribution = Distribution("../img_for_test/Distribution/test2")
        self.assertFalse(balance.isBalanced)
        self.assertTrue(balance.balance_statut == "Empty dataset")
        balance.balance_data()
        self.assertFalse(balance.isBalanced)
        self.assertTrue(balance.balance_statut == "Empty dataset")
        self.assertDictEqual(distribution.getFileCount(), {})

    def test_3_empty_cat(self):
        balance = Balance("../img_for_test/Distribution/test3")
        distribution = Distribution("../img_for_test/Distribution/test3")
        self.assertFalse(balance.isBalanced)
        balance.balance_data()
        self.assertTrue(balance.balance_statut == "Empty category, imposible to balance")
        target = {
            'Apple_Black_rot': 2,
            'Apple_healthy': 0
        }
        self.assertFalse(balance.isBalanced)
        self.assertDictEqual(distribution.getFileCount(), target)

    def test_4_already_balanced(self):
        balance = Balance("../img_for_test/Distribution/test4")
        distribution = Distribution("../img_for_test/Distribution/test4")
        init_distrib = distribution.getFileCount()
        self.assertTrue(balance.isBalanced)
        self.assertTrue(balance.balance_statut == "Balanced")
        balance.balance_data()
        self.assertTrue(balance.balance_statut == "Balanced")
        self.assertTrue(balance.isBalanced)
        final_distrib = distribution.getFileCount()
        self.assertDictEqual(init_distrib, final_distrib)


if __name__ == "__main__":
    unittest.main()