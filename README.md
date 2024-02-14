# Leaffliction

This 42 Machine Learning project aims at testing Convolutional Neural Network for computer vision. 

The goal is to train a program to ba able to recognise photos of leaves with and without deseases. 

The dataset can be download here https://cdn.intra.42.fr/document/document/17547/leaves.zip

## Distribution

The Distribution.py program shows how the dataset is distributed. For example: 

```?> python src/Distribution.py images ```

![Distribution chart](./img/distribution.png "Distribution")

As we can see, we need to balance the dataset by adding more images for some of the categories. 

## Augmentation

The Augmentation.py program takes an image and modify it to create 6 new images. 

```?> python src/Augmentation.py images/Grape_spot/image\ \(1\).JPG```

![Augmentation](./img/Augmentation.png "Augmentation")

The new images are saved in the same directory than the orignal. 

It is now possible to balance the dataset with :

```?> python src/Balance.py images```

It will create enough images to balance the dataset.

![Balanced](./img/Balanced.png "Balanced")

## Transformation

This part aims at transforming the image to get some information about it. We used the plantCV library to do most of the transformation. The 42 subject asked us to create transformation for the dataset that we could use to learn the leaves characteristics. However, we ended up not using them and using a Convolutional Neural Network instead. The CNN learns itself what are the best filters and transformation to learn. 

```?>python src/Transformation.py images/Apple_Black_rot/image\ \(100\).JPG```

![Tranformation](./img/Transformation1.png "Tranformation")
![Tranformation](./img/Transformation2.png "Tranformation")

## Training

The Train.py program uses the TensorFlow library. It create a CNN with the following structure : 

```
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
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
```

It uses Adam as optimiser and SparseCategoricalCrossentropy as a loss function. 

After quite a few epoch, the accuracy was able to reach around 97% for the validation dataset. The model is saved in the Learning.zip file. 

## Predict

The predict program takes an image, predict what it is, and then show the result.

![Tranformation](./img/result.png "result")