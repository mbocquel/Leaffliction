import tensorflow as tf
from .base_model import BaseModel
from dataloader.dataloader import DataLoader
import logging
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


logger = logging.getLogger(__name__)
handler = logging.FileHandler(f"logs/{__name__}.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class SaveModelCallback(tf.keras.callbacks.Callback):
    """
    Callback to save the model after each epoch.

    Args:
        filepath (str): The path to save the model.

    Methods:
        on_epoch_end(epoch, logs=None):
            Called at the end of each epoch.
            Saves the model at the specified filepath.
    """

    def __init__(self, filepath):
        super(SaveModelCallback, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.
        Saves the model at the specified filepath.

        Args:
            epoch (int): The current epoch number.
            logs (dict): Dictionary containing the training metrics for the current epoch.
        """
        self.model.save(self.filepath)
        logger.info(f"Model saved at {self.filepath} after epoch {epoch}")


class My_CNN_model(BaseModel):
    """My CNN Model"""

    def __init__(self, config):
        """
        Initializes the My_CNN_model class.

        Args:
            config (object): Configuration object containing model parameters.
        """
        super().__init__(config)
        self.base_model = tf.keras.applications.VGG16(
            input_shape=self.config.model.input_shape,
            weights="imagenet",
            include_top=False,
        )
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.output_channels = self.config.model.output_channels
        self.batch_size = self.config.train.batch_size
        self.epoches = self.config.train.epoches
        self.optimizer = self.config.train.optimizer
        self.metrics = self.config.train.metrics
        self.image_size = self.config.data.image_size
        self.save_name = self.config.model.save_name
        self.class_names = [f"class_{i}" for i in range(8)]

    def load_data(self):
        """
        Loads and preprocesses the data.
        """
        train_dataset, val_dataset = DataLoader().load_data(
            self.config.data, subset="both"
        )
        self.class_names = train_dataset.class_names
        self.train_dataset = train_dataset.map(
            self._normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.val_dataset = val_dataset.map(self._normalize)
        logger.info(f"Dataset loaded")

    def _normalize(self, input_image, label):
        """
        Normalizes the input image.

        Args:
            input_image (tf.Tensor): Input image tensor.
            label (tf.Tensor): Label tensor.

        Returns:
            tf.Tensor: Normalized input image tensor.
            tf.Tensor: Label tensor.
        """
        input_image = tf.cast(input_image, tf.float32) / 255.0
        return input_image, label

    def build(self):
        """
        Builds the Keras model based on the base model and additional layers.
        """
        for layer in self.base_model.layers:
            layer.trainable = True
        layers = self.base_model.layers
        layers += [
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(
                64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.1)
            ),
            tf.keras.layers.Dense(self.output_channels, activation="softmax"),
        ]
        self.model = tf.keras.models.Sequential(layers)
        self.model.build()
        logger.info(f"Model build")

    def compile(self):
        """
        Compiles the model with loss, optimizer, and metrics.
        """
        if not self.model:
            logger.error(f"The Model is not build, impossible to compile")
            return
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=self.config.train.optimizer,
            metrics=self.config.train.metrics,
        )

    def train(self):
        """
        Compiles and trains the model.

        Returns:
            list: Training loss history.
            list: Validation loss history.
        """
        if not self.model:
            logger.error(f"The Model is not build, impossible to train")
            return
        self.compile()
        save_callback = SaveModelCallback(filepath=self.save_name)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        model_history = self.model.fit(
            self.train_dataset,
            epochs=self.epoches,
            validation_data=self.val_dataset,
            verbose=1,
            callbacks=[save_callback, early_stopping],
        )
        return model_history.history["loss"], model_history.history["val_loss"]

    def evaluate(self):
        """
        Predicts results for the validation dataset.

        Returns:
            dict: Evaluation metrics.
        """
        if not self.val_dataset:
            logger.error(f"There is no validation dataset to use")
            return
        print("Evaluate")
        self.compile()
        if self.model is not None:
            result = self.model.evaluate(self.val_dataset)
            return dict(zip(self.model.metrics_names, result))

    def save(self):
        """
        Saves the model and its parameters.
        """
        if not self.model:
            logger.error(f"The Model is not build, impossible to save")
            return
        self.model.save(self.save_name)
        logger.info(f"Model saved at {self.save_name}")

    def load(self, path=None):
        """
        Loads a model.

        Args:
            path (str, optional): Path to the model file. If not provided, loads from the default save path.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not path and not os.path.isfile(self.save_name):
            logger.error(f"Tried to load a model from a non-existent file")
            return
        elif path:
            self.model = tf.keras.models.load_model(path)
        else:
            self.model = tf.keras.models.load_model(self.save_name)

    def predict(self, path, print=True):
        """
        Predicts the category of an image or a directory.

        Args:
            path (str): Path to the image file or directory.
            print (bool, optional): Whether to print the prediction result. Defaults to True.

        Returns:
            list: Predicted categories.
            list: Confidence scores.
        """
        if not self.model:
            logger.error(f"The Model is not build, impossible to predict")
            raise ValueError("The Model is not build, impossible to predict")
        predictions = []
        confidences = []
        if os.path.isdir(path):
            file_list = []
            for foldername, subdirectorys, filenames in os.walk(path):
                filenames = [
                    os.path.join(foldername, filename)
                    for filename in filenames
                    if filename.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                file_list += filenames
            for file in file_list:
                pred, conf = self.predict_one_img(
                    np.array(Image.open(file, "r")), print=print
                )
                predictions.append(pred)
                confidences.append(conf)
        else:
            pred, conf = self.predict_one_img(
                np.array(Image.open(path, "r")), print=print
            )
            predictions.append(pred)
            confidences.append(conf)
        return predictions, confidences

    def predict_one_img(self, img, print=True):
        """
        Predicts the category of one image and prints a graph if requested.

        Args:
            img (np.ndarray): Input image array.
            print (bool, optional): Whether to print the prediction result. Defaults to True.

        Returns:
            str: Predicted category.
            float: Confidence score.
        """
        if not self.model:
            logger.error(f"The Model is not build, impossible to predict")
            return None, None
        img = Image.fromarray(img).convert("RGB")
        img_resize = np.array(
            img.resize((self.config.data.image_size, self.config.data.image_size))
        )
        img_norm, _ = self._normalize(img_resize, 0)
        y_pred = self.model.predict(
            tf.reshape(img_norm, tuple([1] + self.config.model.input_shape))
        )
        predicted_label = self.class_names[np.argmax(y_pred)]
        confidence = float(np.max(y_pred))
        if print:
            fig = plt.figure()
            fig.patch.set_facecolor("black")
            gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])
            ax0 = fig.add_subplot(gs[0])
            ax0.imshow(img, aspect="equal")
            ax0.axis("off")
            txt_part1 = "===  Leaffliction classification  ==="
            txt_part2 = "Class predicted : " + predicted_label
            ax2 = fig.add_subplot(gs[1])
            ax2.set_facecolor("black")
            ax2.text(
                0.5,
                0.7,
                txt_part1,
                color="white",
                fontsize=25,
                ha="center",
                va="center",
            )
            ax2.text(
                0.5,
                0.4,
                txt_part2,
                color="white",
                fontsize=14,
                ha="center",
                va="center",
            )
            plt.subplots_adjust(
                left=0.02, right=0.98, top=0.98, wspace=0.05, hspace=0.2
            )
            manager = plt.get_current_fig_manager()
            if manager is not None:
                manager.set_window_title("Leafflction prediction result")
            plt.show()
        return predicted_label, confidence
