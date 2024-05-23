import unittest
from model.my_CNN_model import My_CNN_model
from utils.config import Config
import tempfile
import random
import numpy as np
from PIL import Image
import os

class TestMyCNNModel(unittest.TestCase):
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
            "train": {
                "batch_size": 32,
                "epoches": 2,
                "optimizer": "adam",
                "metrics": ['accuracy']
            },
            "model": {
                "input_shape": [256, 256, 3],
                "output_channels": 8,
                "save_name": "my_cnn_model.keras"
            }
        }     
        self.config = Config.from_json(CFG)
        self.model = My_CNN_model(CFG)

    def test_load_data(self):
        """
        Check that we were able to load the model 
        and that we have a train and val dataset that is normilised
        """
        self.model.load_data()
        self.assertIsNotNone(self.model.train_dataset)
        self.assertIsNotNone(self.model.val_dataset)
        for img, _ in self.model.train_dataset:
            self.assertTrue((img >= 0).numpy().all())
            self.assertTrue((img <= 1).numpy().all())

    def test_build(self):
        self.model.build()
        self.assertIsNotNone(self.model.model)
        self.assertEqual(len(self.model.model.layers), 21)
        self.assertEqual(self.model.model.layers[-2].units, 64) # 64 units for the second last layer
        self.assertEqual(self.model.model.layers[-1].units, self.config.model.output_channels) #  self.config.model.output_channels units for the last layer
        self.assertTrue(self.model.model.layers[-2].trainable)
        self.assertTrue(self.model.model.layers[-1].trainable)


    def test_train(self):
        self.model.load_data()
        self.model.build()
        train_loss, val_loss = self.model.train()
        self.assertIsNotNone(train_loss)
        self.assertIsNotNone(val_loss)

    def test_evaluate(self):
        self.model.load_data()
        self.model.build()
        self.model.compile()
        predictions = self.model.evaluate()
        self.assertIsNotNone(predictions)

    def test_predict_dir(self):
        self.model.build()
        self.test_dir = tempfile.TemporaryDirectory()
        for i in range(5):  # Creation of 5 images for test
            image_path = os.path.join(self.test_dir.name, f'test_image_{i}.png')
            image = Image.new('RGB', (256, 256), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            image.save(image_path)
        predictions, confidences = self.model.predict(self.test_dir.name, print=False)
        self.assertEqual(len(predictions), 5)
        self.assertEqual(len(confidences), 5)
        self.assertTrue(np.array([type(conf) == np.float32 for conf in confidences]).all())
        self.assertTrue(np.array([type(predict) == str for predict in predictions]).all())
        self.test_dir.cleanup()

    def test_predict_solo(self):
        self.model.build()
        self.test_dir = tempfile.TemporaryDirectory()
        image_path = os.path.join(self.test_dir.name, f'test_image.png')
        image = Image.new('RGB', (256, 256), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        image.save(image_path)
        prediction, confidence = self.model.predict(image_path, print=False)
        self.assertEqual(len(prediction), 1)
        self.assertTrue(type(prediction[0]) == str)
        self.assertEqual(len(confidence), 1)
        self.assertTrue(type(confidence[0]) == np.float32)
        self.test_dir.cleanup()


if __name__ == '__main__':
    unittest.main()

