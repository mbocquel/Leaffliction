import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from abc import ABC
import argparse

class Augmentation(ABC):
    """
    Class that augment an image by creating new transformed images from it
    """
    def __init__(self, path, rot_angle=30, illum_level=1.5, zoom=5, cont_factor=1.5) -> None:
        super().__init__()
        self.img_path = path
        self.img = Image.open(self.img_path)
        self.param = {
            "rot_angle":rot_angle,
            "illum_level": illum_level,
            "zoom": zoom,
            "cont_factor":cont_factor
        }
        self.aug_img = {
            "rotated": None, 
            "fliped": None,
            "blured": None,
            "illuminated": None,
            "scaled": None, 
            "contrasted": None
        }
    
    def __del__(self):
        """Class Destructor"""
        self.img.close()
        for img in self.aug_img.values():
            if img:
                img.close()

    def save_img_aug(self, type):
        """Save the augmented images"""
        if not self.aug_img[type]:
            return
        filepath_split = os.path.splitext(self.img_path)
        new_path = filepath_split[0] + "_"
        match type:
            case 'rotated':
                new_path += type + "_" + str(self.param["rot_angle"]) + ".png"
            case 'illuminated':
                new_path += type + "_" + str(self.param["illum_level"]) + ".png"
            case 'scaled':
                new_path += type + "_" + str(self.param["zoom"]) + ".png"
            case 'contrasted':
                new_path += type + "_" + str(self.param["cont_factor"]) + ".png"
            case _:
                new_path += type + ".png"
        self.aug_img[type].save(new_path, format="PNG")

    def rotate_img(self, rot_angle=None, save=True):
        """Rotate the image"""
        if rot_angle:
            self.param["rot_angle"] = rot_angle
        self.aug_img["rotated"] = self.img.rotate(self.param["rot_angle"], fillcolor="#FFFFFF")
        if save:
            self.save_img_aug("rotated")
        return self.aug_img["rotated"]

    def flip_img(self, save=True):
        """Flip the image"""
        self.aug_img["fliped"] = ImageOps.flip(self.img)
        if save:
            self.save_img_aug("fliped")
        return self.aug_img["fliped"]
    
    def blur_img(self, save=True):
        """Blur the image"""
        self.aug_img["blured"] = self.img.filter(ImageFilter.BLUR)
        if save:
            self.save_img_aug("blured")
        return self.aug_img["blured"]

    def illuminate_img(self, illum_level=None, save=True):
        """illuminate the image"""
        if illum_level:
            self.param["illum_level"] = illum_level
        self.aug_img["illuminated"] = ImageEnhance.Brightness(self.img).enhance(self.param["illum_level"])
        if save:
            self.save_img_aug("illuminated")
        return self.aug_img["illuminated"]

    def scale_img(self, zoom=None, save=True):
        """Scale the image"""
        if zoom:
            self.param["zoom"] = zoom
        w, h = self.img.size
        self.aug_img["scaled"] = self.img
        if self.param["zoom"] and self.param["zoom"] != 1:
            img_crop = self.img.crop(((w // 2) - w / self.param["zoom"], (h // 2) - h / self.param["zoom"],
                            (w // 2) + w / self.param["zoom"], (h // 2) + h / self.param["zoom"]))
            self.aug_img["scaled"] = img_crop.resize((w, h), Image.LANCZOS)
        if save:
            self.save_img_aug("scaled")
        return self.aug_img["scaled"]

    def increase_contrast(self, cont_factor=None, save=True):
        """Increase contrast"""
        if cont_factor:
            self.param["cont_factor"] = cont_factor
        self.aug_img["contrasted"] = ImageEnhance.Contrast(self.img).enhance(self.param["cont_factor"])
        if save:
            self.save_img_aug("contrasted")
        return self.aug_img["contrasted"]
    
    def generate_augmented_imgs(self):
        """Generate all the transformed image"""
        self.rotate_img()
        self.flip_img()
        self.blur_img()
        self.illuminate_img()
        self.scale_img()
        self.increase_contrast()
    
    def plot_img(self):
        """Plot all generated images"""
        nb_img = 1 + len([1 for val in self.aug_img.values() if val])
        cur_img = 1
        fig = plt.figure(figsize=(2.1 * nb_img , 2))
        fig.add_subplot(1, nb_img, cur_img)
        plt.imshow(self.img)
        plt.axis('off')
        plt.title("original")
        for key, img in self.aug_img.items():
            if img:
                cur_img +=1
                fig.add_subplot(1, nb_img, cur_img)
                plt.imshow(img)
                plt.axis('off')
                plt.title(key)
        plt.tight_layout()
        manager = plt.get_current_fig_manager()
        manager.set_window_title("Augmentation for " + self.img_path)
        plt.show()


def process_arg(**kwargs):
    """Process the program input arguments"""
    plot = kwargs["show"]
    args = {}
    for key, val in kwargs.items():
        if key == "show" or not val:
            continue
        args[key] = val
    assert "path" in args and os.path.isfile(args["path"]), "Please provide a valid path"
    return args, plot


def main(**kwargs):
    try:
        args, plot = process_arg(**kwargs)
        augmentation = Augmentation(**args)
        augmentation.generate_augmented_imgs()
        if plot:
            augmentation.plot_img()

    except Exception as err:
        print("Error: ", err)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Description of the dataset")
    parser.add_argument("--path", "-p", type=str,
                        help="Path to the image to augment")
    parser.add_argument("--show", "-s", action="store_true",
                        help="Show the plot")
    parser.add_argument("--rot_angle", "-ra", type=float,
                        help="Rotation angle")
    parser.add_argument("--illum_level", "-il", type=float,
                        help="Illumination level")
    parser.add_argument("--zoom", "-z", type=float,
                        help="Zoom level")
    parser.add_argument("--cont_factor", "-cf", type=float,
                        help="Contrast Factor")
    args = parser.parse_args()
    kwargs = {key: getattr(args, key) for key in vars(args)}
    main(**kwargs)
