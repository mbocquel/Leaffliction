from abc import ABC
from Distribution import Distribution
from Augmentation import Augmentation
import random
import os
import argparse

class Balance(ABC):
    """
    Class that can balance the data set by increasing the number of images through augmentation
    """
    def __init__(self, dir) -> None:
        """
        Class constructor
        """
        super().__init__()
        self.imageDir = dir
        self.distribution = Distribution(dir)
        self.buffer = 1
        self.checkIfBalanced()

    def checkIfBalanced(self):
        file_count = self.distribution.getFileCount()
        nb_files = sum(file_count.values())
        if nb_files == 0:
            self.isBalanced = False
            self.balance_statut = "Empty dataset"
            return
        if min(file_count.values()) == 0:
            self.isBalanced = False
            self.balance_statut = "Empty category, imposible to balance"
            return
        val_percent = [(val *100) // nb_files for val in  file_count.values()]
        self.isBalanced = (max(val_percent) - min(val_percent) <= self.buffer)
        if self.isBalanced:
            self.balance_statut = "Balanced"
        else:
            self.balance_statut = "Not Balanced"
    
    def balance_data(self):
        """
        Balance the dataset
        """
        if self.balance_statut == "Empty dataset":
            return
        file_count = self.distribution.getFileCount()
        max_file = max(file_count.values())
        max_categories = [key for key, val in file_count.items() if val == max_file]
        categories_to_balance = [key for key, val in file_count.items() if key not in max_categories and val > 0 ]
        for cat in categories_to_balance:
            self.balance_category(cat, max_file)
        self.checkIfBalanced()
        
    def balance_category(self, cat, max_file):
        """
        Increase the number of images in a specific category to reach max_file images
        """
        files = list(self.distribution.getFileList()[cat])
        nb_files_to_create = max_file - len(files)
        transfo_to_do = set()
        for i in range(nb_files_to_create):
            selection = self.get_combinaison_transfo(files)
            while selection in transfo_to_do:
                selection = self.get_combinaison_transfo(files)
            transfo_to_do.add(selection)
        print(f"Creating {len(transfo_to_do)} new images for {cat}")
        for elem in transfo_to_do:
            file, transfo, param = elem
            aug = Augmentation(file)
            match transfo:
                case 'rotated':
                    aug.rotate_img(rot_angle=param)
                case 'fliped':
                    aug.flip_img()
                case 'blured':
                    aug.blur_img()
                case 'illuminated':
                    aug.illuminate_img(illum_level=param)            
                case 'scaled':
                    aug.scale_img(zoom=param)
                case 'contrasted':
                    aug.increase_contrast(cont_factor=param)            

    def get_combinaison_transfo(self, files):
        possible_transfo = ["rotated", "fliped", "blured", "illuminated", "scaled", "contrasted"]
        selection = [random.choice(files), random.choice(possible_transfo), None]
        if selection[1] == "rotated":
            selection[2] =  10 + int(random.random() * 35) # Angle from 10 to 45 degrees
        if selection[1] == "illuminated":
            selection[2] =  round(0.5 + random.random() * 2, 2) # illum factor from 0.5 to 2.5
        if selection[1] == "scaled":
            selection[2] =  2 + int(random.random() * 8) # zoom from 2 to 10
        if selection[1] == "contrasted":
            selection[2] =  round(0.5 + random.random() * 2, 2) # illum factor from 0.5 to 2.5
        return tuple(selection)
    
    def remove_augmented_img(self):
        files_by_cat = self.distribution.getFileList()
        file_to_remove = []
        for values in files_by_cat.values():   
            tmp = [[file, file.split("/")[-1]] for file in list(values)]
            file_to_remove += [file[0] for file in tmp if "_" in file[1]]
        for file in file_to_remove:
            os.remove(file)


def main(**kwargs):
    try:
        path = kwargs["path"]
        assert os.path.isdir(path), "Please provide a valid path for the directory"
        balance = Balance(path)
        if kwargs["balance"]:
            balance.balance_data()
        if kwargs["clean"]:
            balance.remove_augmented_img()

    except Exception as err:
        print("Error: ", err)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Description of the dataset")
    parser.add_argument("--path", "-p", type=str,
                        help="Path to the directory to balance or clean")
    parser.add_argument("--balance", "-b", action="store_true",
                        help="Balance the directory")
    parser.add_argument("--clean", "-c", action="store_true",
                        help="Clean the directory by removing the augmented images")
    args = parser.parse_args()
    kwargs = {key: getattr(args, key) for key in vars(args)}
    main(**kwargs)