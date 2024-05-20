from unittest.mock import patch
from model.my_CNN_model import My_CNN_model
from configs.config import CFG

import tensorflow as tf
import  numpy as np
import tensorflow_datasets as tfds


def dummy_load_data(*args, **kwargs):
    with tfds.testing.mock_data(num_examples=1):
        return tfds.load(CFG['data']['path'], with_info=True)


class MyModelTest(tf.test.TestCase):

    def setUp(self):
        super(MyModelTest, self).setUp()
        self.my_model = My_CNN_model(CFG)

    def tearDown(self):
        pass

    def test_normalize(self):
        input_image = np.array([[1., 1.], [1., 1.]])
        input_mask = 1
        expected_image = np.array([[0.00392157, 0.00392157], [0.00392157, 0.00392157]])

        result = self.my_model._normalize(input_image, input_mask)
        self.assertAllClose(expected_image, result[0])

    # def test_ouput_size(self):
    #     shape = (1, self.unet.image_size, self.unet.image_size, 3)
    #     image = tf.ones(shape)
    #     self.unet.build()
    #     self.assertEqual(self.unet.model.predict(image).shape, shape)

    @patch('model.unet.DataLoader.load_data')
    def test_load_data(self, mock_data_loader):
        mock_data_loader.side_effect = dummy_load_data
        shape = tf.TensorShape([None, self.my_model.image_size, self.my_model.image_size, 3])

        self.my_model.load_data()
        mock_data_loader.assert_called()
        

        # self.assertItemsEqual(self.my_model.train_dataset.element_spec[0].shape, shape)
        # self.assertItemsEqual(self.my_model.test_dataset.element_spec[0].shape, shape)



if __name__ == '__main__':
    tf.test.main()


# coverage run -m unittest /home/aisummer/PycharmProjects/Deep-Learning-Production-Course/model/tests/unet_test.py
# coverage report -m  /home/aisummer/PycharmProjects/Deep-Learning-Production-Course/model/tests/unet_test.py