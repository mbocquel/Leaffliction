import argparse
import os
from model.my_CNN_model import My_CNN_model
from dataloader.Augmentation import Augmentation
from dataloader.Distribution import Distribution
from dataloader.Balance import Balance
from configs.CFG import CFG
from utils.config import Config
import logging


logger = logging.getLogger(__name__)
handler = logging.FileHandler(f"logs/{__name__}.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def train(model:My_CNN_model):
    """Initiate a new model, train it and return it"""
    if not model.train_dataset:
        model.load_data()
    if not model.model:
        model.build()
    train_loss, val_loss = model.train()
    

def predict(model:My_CNN_model, path:str, show_result=True):
    """Use a model to make a prediction"""
    model.predict(path=path, print=show_result)


def main(**kwargs) -> int:
    try:
        config = Config.from_json(CFG)
        model = None
        if kwargs.get("distribution"):
            Distribution(config.data.path).plot()

        if kwargs.get("augmentation"):
            Augmentation(**CFG["augmentation"]).generate_augmented_imgs()
            
        if kwargs.get("balance"):
            Balance(config.data.path).balance_data()
        
        if kwargs.get("load") and os.path.isfile(config.model.save_name):
            model = My_CNN_model(CFG)
            model.load(config.model.save_name)
        elif kwargs.get("load"):
            logger.error("No available model to load")
            return
        
        if kwargs.get("train"):
            if not model:
                model = My_CNN_model(CFG)
            train(model)

        if kwargs.get("predict"):
            model = My_CNN_model(CFG)
            model.load(config.predict.model_path)
            model.class_names = config.predict.class_names
            predict(model, config.predict.img_path, show_result=config.predict.show_result_plot)
        
        if kwargs.get("clean"):
            Balance(config.data.path).remove_augmented_img()
        
    except Exception as err:
        print("Error:", err)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Description of the dataset")
    parser.add_argument("--distribution", "-d", action="store_true",
                        help="Launch Distribution part of the program")
    parser.add_argument("--augmentation", "-a", action="store_true",
                        help="Launch Augmentation part of the program")
    parser.add_argument("--balance", "-b", action="store_true",
                        help="Launch Balance part of the program")
    parser.add_argument("--train", "-t", action="store_true",
                        help="Launch Train part of the program")
    parser.add_argument("--predict", "-p", action="store_true",
                        help="Launch Predict part of the program")
    parser.add_argument("--clean", "-c", action="store_true",
                        help="Clean the directory by removing the augmented images")
    parser.add_argument("--load", "-l", action="store_true",
                        help="Load an existing model")
    args = parser.parse_args()
    kwargs = {key: getattr(args, key) for key in vars(args)}
    main(**kwargs)
