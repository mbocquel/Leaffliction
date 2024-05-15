import os
import matplotlib.pyplot as plt
from abc import ABC
import argparse

class Distribution(ABC):
    """
    Class that create the distribution graph of all images from a directory
    """
    def __init__(self, dir) -> None:
        super().__init__()
        self.imageDir = dir
        self.file_count = self.computeFileCount()

    def computeFileCount(self):
        """Create a dict with the categories and their number of images"""
        data = {}
        for folder in os.listdir(self.imageDir):
            data[folder] = 0
            for foldername, subdirectorys, filenames in os.walk(self.imageDir + "/" + folder):
                filenames = [filename for filename in filenames if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
                data[folder] += len(filenames)
        return data

    def getFileCount(self):
        """Return the file count dict"""
        return self.file_count.copy()
    
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
        path = None
        for key, value in kwargs.items():
            if key == "path":
                path = value
        assert os.path.isdir(path), "Please enter a correct path"
        distribution = Distribution(path)
        distribution.plot()

    except Exception as err:
        print("Error: ", err)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Description of the dataset")
    parser.add_argument("--path", "-p", type=str,
                        help="Path to the dataset")
    args = parser.parse_args()
    kwargs = {key: getattr(args, key) for key in vars(args)}
    main(**kwargs)
