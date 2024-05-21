import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(config_data, subset="both"):
        """Loads a dataset from a directory and split it into train and validation sets."""
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            config_data.path,
            labels='inferred',
            label_mode='int',
            class_names=None,
            shuffle=config_data.path,
            image_size=(config_data.image_size, config_data.image_size),
            batch_size=config_data.batch_size,
            seed=config_data.seed,
            validation_split=config_data.val_split,
            subset=subset
        )
        train, val = ds
        class_names = train.class_names
        num_classes = len(class_names)
        logger.info(f"Found {num_classes} classes: {class_names}")

        num_batches = tf.data.experimental.cardinality(train).numpy()
        logger.info(f"Number of batches: {num_batches}")

        return train, val
