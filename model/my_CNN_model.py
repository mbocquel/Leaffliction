import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping
from .base_model import BaseModel
from dataloader.dataloader import DataLoader
import logging

logger = logging.getLogger(__name__)

class SaveModelCallback(Callback):
    def __init__(self, filepath):
        super(SaveModelCallback, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.filepath)
        logger.info(f"Model saved at {self.filepath} after epoch {epoch}")


class My_CNN_model(BaseModel):
    """My CNN Model"""
    def __init__(self, config):
        super().__init__(config)
        self.base_model = tf.keras.applications.VGG16(input_shape=self.config.model.input_shape, weights='imagenet', include_top=False)
        self.model = None
        self.output_channels = self.config.model.output_channels
        self.batch_size = self.config.train.batch_size
        self.epoches = self.config.train.epoches
        self.optimizer = self.config.train.optimizer
        self.metrics = self.config.train.metrics
        self.image_size = self.config.data.image_size
        self.save_name = self.config.model.save_name

    def load_data(self):
        """Loads and Preprocess data """
        train_dataset, val_dataset = DataLoader().load_data(self.config.data, subset="both")
        self.class_names = train_dataset.class_names
        self.train_dataset = train_dataset.map(self._normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.val_dataset= val_dataset.map(self._normalize)

    def _normalize(self, input_image, label):
        """Normalise input image"""
        input_image = tf.cast(input_image, tf.float32) / 255.0
        return input_image, label

    def build(self):
        """Builds the Keras model based"""
        for layer in self.base_model.layers:
            layer.trainable = False
        layers = self.base_model.layers
        layers += [
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
                tf.keras.layers.Dense(self.output_channels, activation='softmax')
            ]
        self.model = tf.keras.models.Sequential(layers)
        self.model.build(input_shape=self.config.model.input_shape)

    def train(self):
        """Compiles and trains the model"""
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=self.config.train.optimizer,
            metrics=self.config.train.metrics)
        save_callback = SaveModelCallback(filepath=self.save_name)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_history = self.model.fit(self.train_dataset,
                                       epochs=self.epoches,
                                       validation_data=self.val_dataset,
                                       verbose=1,
                                       callbacks=[save_callback, early_stopping])
        return model_history.history['loss'], model_history.history['val_loss']

    def evaluate(self):
        """Predicts results for the validation dataset"""
        pass

    def save(self):
        """Save the model and it's parameters"""
        self.model.save(self.save_name)
        logger.info(f"Model saved at {self.save_name}")

    def predict(self, img):
        """Predict the category of an image"""
        img, label = self._normalize(img, 0)
        # predicted_label =

