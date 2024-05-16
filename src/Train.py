import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import regularizers
from Balance import balance
from Predict import extract_file_from_zip
import os
import argparse
import zipfile
import shutil
import random
from abc import ABC

class SaveModelCallback(Callback):
    def __init__(self, filepath):
        super(SaveModelCallback, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.filepath)


class MyLeafCNN(ABC):
    """"""


class Utils:
    """Class that store some usefull methods to manage the images"""

def normalise_img(image, label):
    image = tf.cast(image/255.0, tf.float32)
    return image, label


def loadDataset(path, img_size, batch):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        shuffle=True,
        image_size=(img_size, img_size),
        batch_size=batch
    )
    return ds


def getDsPartitionTf(ds, train_size, val_size):
    if (os.path.isdir("train.tfrecord") and os.path.isdir("val.tfrecord")):
        train_dataset = tf.data.Dataset.load("train.tfrecord")
        cv_dataset = tf.data.Dataset.load("val.tfrecord")
        print(f"-Train dataset loaded with {len(train_dataset)*32} images")
        print(f"-Validation dataset loaded with {len(cv_dataset)*32} images")
    else:
        train_split = 0.70
        shuffle_size = 100
        ds = ds.shuffle(shuffle_size, seed=12)
        if train_size is not None:
            len_train = int(train_size / 32.0) + 1
        else:
            len_train = int(len(ds) * train_split)
        if val_size is not None:
            len_val_ds = int(val_size / 32.0) + 1
        else:
            len_val_ds = int(((1 - train_split) / train_split) * len_train)
        train_dataset = ds.take(len_train)
        print(f"-Train dataset created with {len(train_dataset)*32} images")
        cv_dataset = ds.skip(len_train).take(len_val_ds)
        print(f"-Validation dataset created with {len(cv_dataset)*32} images")
        # tf.data.Dataset.save(train_dataset, "train.tfrecord")
        # tf.data.Dataset.save(cv_dataset, "val.tfrecord")
    return train_dataset, cv_dataset


def createFinalZip(zipFileName):
    learningFilePath = "model_param.keras"
    classNamesCsv = "class_names.csv"
    with zipfile.ZipFile(zipFileName, 'w') as zipf:
        zipf.write(learningFilePath)
        zipf.write(classNamesCsv)


def processArgs(**kwargs):
    epochs = 15
    path = None
    save_name = "Learning"
    train_size = None
    val_size = None
    for key, value in kwargs.items():
        if value is not None:
            match key:
                case 'epochs':
                    epochs = value
                case 'path':
                    path = value
                case 'save_name':
                    save_name = value
                case 'train_size':
                    train_size = value
                case 'val_size':
                    val_size = value
    return epochs, path, save_name, train_size, val_size


def get_reduced_files(path, train_size, val_size):
    img_path_list = [
         [[foldername, fn, '/'.join(
              [e for e in foldername.split("/") if e not in ["..", "."]])]
          for fn in filenames]
         for foldername, subdirectory, filenames in os.walk(path)
         if len(filenames)]
    img_path_list = np.array([element for sous_liste in
                              img_path_list for element in sous_liste])
    list_path_long = list(set([img[2] for img in img_path_list]))
    img_path_list = [[img[0], img[1], img[2].replace(
        os.path.commonpath(list_path_long) + '/', '')]
            for img in img_path_list]
    random.shuffle(img_path_list)
    selected_img = img_path_list[:(train_size + val_size)]
    [os.makedirs("train_tmp/" + path, exist_ok=True)
     for path in list(set([img[2] for img in img_path_list]))]
    for img in selected_img:
        shutil.copy2(img[0] + "/" + img[1], "train_tmp/" +
                     img[2] + "/" + img[1])


def main(**kwargs):
    try:
        print("\n")
        epochs, path, saveN, train_size, val_size = processArgs(**kwargs)
        assert path is not None, "Please enter a directory path as parametter"
        assert os.path.isdir(path), "Please enter a directory as a parametter"
        if (train_size is not None and val_size is None):
            raise AssertionError("You need to define both train and val size")
        if (train_size is None and val_size is not None):
            raise AssertionError("You need to define both train and val size")
        imgSize = 256
        input_shape = (imgSize, imgSize, 3)
        batch = 32
        if train_size is not None:
            get_reduced_files(path, train_size, val_size)
            path = "train_tmp"
        print("Balancing the dataset.................................")
        balance(path)
        print("......................................................done !\n")

        print("Loading dataset.......................................")
        ds = loadDataset(path, imgSize, batch)
        class_names = ds.class_names
        ds = ds.map(normalise_img)
        train_ds, validation_ds = getDsPartitionTf(ds, train_size, val_size)
        np.savetxt("class_names.csv", class_names, delimiter=',', fmt='%s')
        print("......................................................done !\n")

        if os.path.isfile("model_param.keras"):
            print("Loading CNN model....................................")
            model = load_model('model_param.keras')
            print("................................................done !\n")
        elif os.path.isfile("Learning.zip"):
            print("Loading CNN model....................................")
            extract_file_from_zip("Learning.zip", "model_param.keras", ".")
            model = load_model('model_param.keras')
            print("................................................done !\n")
        else:
            # CNN Model definition
            print("Defining CNN model....................................")
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(64, activation='relu',
                      kernel_regularizer=regularizers.l2(0.1)),
                Dense(len(class_names), activation='softmax')
            ])
            model.build(input_shape=input_shape)
            model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=False),
                optimizer='adam',
                metrics=['accuracy'])
            print("................................................done !\n")
        # CNN Learning
        print("Learning phase........................................")
        save_callback = SaveModelCallback(filepath='model_param.keras')
        model.fit(
            train_ds,
            epochs=epochs,
            batch_size=batch,
            verbose=1,
            validation_data=validation_ds,
            callbacks=[save_callback]
        )
        print("......................................................done !\n")

        # Saving the model
        print("Saving the model......................................")
        model.save('model_param.keras')
        print("......................................................done !\n")

        # Creation du zip avec les learning et les images
        print("Creating Learning.zip.................................")
        createFinalZip(saveN + '.zip')
        print("......................................................done !\n")

        # Suppression des fichiers innutiles
        print("Removing tmp files....................................")
        if os.path.isdir("train_tmp"):
            shutil.rmtree("train_tmp")
        shutil.rmtree('train.tfrecord')
        shutil.rmtree('val.tfrecord')
        os.remove('model_param.keras')
        os.remove('class_names.csv')
        print("......................................................done !\n")

    except Exception as err:
        print("Error: ", err)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training program for Leaffliction")
    parser.add_argument("--epochs", "-e", type=int,
                        help="Number of epochs for the training")
    parser.add_argument("--path", "-p", type=str,
                        help="Path to the dataset directory")
    parser.add_argument("--save_name", "-sn", type=str,
                        help="Name of the learning saving file")
    parser.add_argument("--train_size", "-ts", type=int,
                        help="Size of the training dataset")
    parser.add_argument("--val_size", "-vs", type=int,
                        help="Size of the validation dataset")

    args = parser.parse_args()
    kwargs = {key: getattr(args, key) for key in vars(args)}
    main(**kwargs)
