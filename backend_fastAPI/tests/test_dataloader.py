import unittest
from dataloader.dataloader import DataLoader
from utils.config import Config

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        CFG = {
            "data": {
                "path": "img_for_test/Distribution/test1",
                "image_size": 256,
                "load_with_info": True,
                "shuffle":True,
                "seed":42,
                "val_split":0.2,
                "batch_size": 32,
            },
            "train": {},
            "model": {}
        }
        self.config = Config.from_json(CFG)
        self.train_dataset, self.val_dataset = DataLoader.load_data(self.config.data, subset="both")

    def test_load_data(self):
        """Test that we get datasets"""
        self.assertIsNotNone(self.train_dataset)
        self.assertIsNotNone(self.train_dataset)
    
    def test_categories(self):
        """ Test that we get categories """
        self.assertEqual(len(self.train_dataset.class_names), 3)

        

if __name__ == '__main__':
    unittest.main()