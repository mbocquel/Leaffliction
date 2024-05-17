import os
import matplotlib.pyplot as plt
from abc import ABC
import argparse

class Distribution(ABC):
    """
    Class that create the distribution graph of all images from a directory
    Attributes:
        imageDir (str): The directory containing the images.
        file_count (dict): Dictionary storing the count of images per category.
        file_list (dict): Dictionary storing the list of images per category.
    """
    def __init__(self, dir) -> None:
        super().__init__()
        self.imageDir = dir
        self.computeFileCount()

    def computeFileCount(self):
        """Create a dict with the categories and their number of images"""
        self.file_count = {}
        self.file_list = {}
        for folder in os.listdir(self.imageDir):
            if not os.path.isdir(os.path.join(self.imageDir, folder)):
                continue
            self.file_count[folder] = 0
            self.file_list[folder] = []
            for foldername, subdirectorys, filenames in os.walk(os.path.join(self.imageDir, folder)):
                filenames = [os.path.join(foldername, filename) for filename in filenames if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
                self.file_count[folder] += len(filenames)
                self.file_list[folder] += filenames
            self.file_list[folder] = set(self.file_list[folder])

    def getFileCount(self, reload=False):
        """Return the file count dict"""
        if reload:
            self.computeFileCount()
        return self.file_count
    
    def getFileList(self, reload=False):
        """Return the file count dict"""
        if reload:
            self.computeFileCount()
        return self.file_list
    
    def plot(self):
        """Plot the distribution graph """
        plt.figure(figsize=(12, 6))

        # Plot bar chart
        plt.subplot(1, 2, 1)
        plt.bar(self.file_count.keys(), self.file_count.values(), color='lightcoral')
        plt.xticks(rotation=90)
        plt.title('File Count per Directory (Bar Chart)')

        # Plot pie chart
        plt.subplot(1, 2, 2)
        plt.pie(self.file_count.values(), labels=self.file_count.keys(), autopct='%1.1f%%', startangle=90,
                colors=['lightgreen', 'lightcoral', 'skyblue'])
        plt.title('File Count Distribution (Pie Chart)')

        plt.tight_layout()
        manager = plt.get_current_fig_manager()
        manager.set_window_title("Leaffliction distribution")
        plt.show()


def main(**kwargs):
    try:
        path = kwargs.get("path")
        assert os.path.isdir(path), "Please enter a correct path"
        distribution = Distribution(path)
        distribution.plot()

    except (AssertionError, FileNotFoundError, NotADirectoryError) as err:
        print("Error:", err)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Description of the dataset")
    parser.add_argument("--path", "-p", type=str,
                        help="Path to the dataset")
    args = parser.parse_args()
    kwargs = {key: getattr(args, key) for key in vars(args)}
    main(**kwargs)
