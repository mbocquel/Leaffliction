import unittest
from dataloader.Distribution import Distribution
import os


class TestDistribution(unittest.TestCase):
    """Test the distribution class"""

    def setUp(self):
        self.test_dir_1 = os.path.join("img_for_test", "Distribution", "test1")
        self.test_dir_2 = os.path.join("img_for_test", "Distribution", "test2")
        self.test_dir_3 = os.path.join("img_for_test", "Distribution", "test3")

    def test_getFileCount_correctCounts(self):
        distribution = Distribution(self.test_dir_1)
        file_count = distribution.getFileCount()
        self.assertIsInstance(file_count, dict)
        target = {"Apple_Black_rot": 3, "Apple_healthy": 4, "Grape_Esca": 8}
        self.assertDictEqual(file_count, target)

    def test_getFileList_correctList(self):
        distribution = Distribution(self.test_dir_1)
        file_list = distribution.getFileList()
        self.assertIsInstance(file_list, dict)
        target = {
            "Apple_Black_rot": set(
                [
                    os.path.join(self.test_dir_1, "Apple_Black_rot", "image (1).JPG"),
                    os.path.join(self.test_dir_1, "Apple_Black_rot", "image (2).JPG"),
                    os.path.join(self.test_dir_1, "Apple_Black_rot", "image (4).JPG"),
                ]
            ),
            "Apple_healthy": set(
                [
                    os.path.join(self.test_dir_1, "Apple_healthy", "image (1).JPG"),
                    os.path.join(
                        self.test_dir_1, "Apple_healthy", "sub", "image (4).JPG"
                    ),
                    os.path.join(self.test_dir_1, "Apple_healthy", "image (2).JPG"),
                    os.path.join(self.test_dir_1, "Apple_healthy", "image (3).JPG"),
                ]
            ),
            "Grape_Esca": set(
                [
                    os.path.join(self.test_dir_1, "Grape_Esca", "image (12).JPG"),
                    os.path.join(self.test_dir_1, "Grape_Esca", "image (13).JPG"),
                    os.path.join(self.test_dir_1, "Grape_Esca", "image (14).JPG"),
                    os.path.join(self.test_dir_1, "Grape_Esca", "image (15).JPG"),
                    os.path.join(self.test_dir_1, "Grape_Esca", "image (16).JPG"),
                    os.path.join(self.test_dir_1, "Grape_Esca", "image (17).JPG"),
                    os.path.join(self.test_dir_1, "Grape_Esca", "image (18).JPG"),
                    os.path.join(self.test_dir_1, "Grape_Esca", "image (19).JPG"),
                ]
            ),
        }
        self.assertDictEqual(file_list, target)

    def test_getFileCount_emptyDirectory(self):
        distribution = Distribution(self.test_dir_2)
        file_count = distribution.getFileCount()
        self.assertIsInstance(file_count, dict)
        target = {}
        self.assertDictEqual(file_count, target)

    def test_getFileCount_emptyCategory(self):
        distribution = Distribution(self.test_dir_3)
        data = distribution.getFileCount()
        self.assertIsInstance(data, dict)
        target = {"Apple_Black_rot": 2, "Apple_healthy": 0}
        self.assertDictEqual(data, target)


if __name__ == "__main__":
    unittest.main()
