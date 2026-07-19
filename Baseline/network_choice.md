In the fine-grained image classification task of recognizing cat breeds (Pallas's cat, Persian cat, Ragdoll, Singapura, Sphynx), it requires distinguishing highly similar subcategories within the cat family. Therefore, the trained model must have the capability to capture fine-grained features that differentiate various targets.

We choose ConvNeXt as the backbone network based on the following considerations.

1. ConvNeXt adopts a multi-stage hierarchical structure, with each stage outputting feature maps at different resolutions. Its design incorporates depthwise separable convolutions, inverted bottleneck structures, and 7×7 convolution kernels, which outperform traditional ResNet models at computational efficiency and feature representation capability levels.
2. ConvNeXt is easier to train on small to medium-sized datasets and is less prone to overfitting, making it well-suited for the requirements of our task.
3. With the parameter scale comparable to ResNet-50, ConvNeXt supports a higher input image resolution (384×384), which helps preserve more detailed information. This leads to better performance when distinguishing target cats with similar coat colors or facial features.

Our baseline project uses the ConvNeXt-Small model for transfer learning. The training process comprises of two stages: feature extraction and fine-tuning.

In the feature extraction stage, all convolutional layer parameters in the backbone network are frozen, and only the subsequent 2 fully connected layers and the 5-class classification head are trained. By employing two fully connected layers, the feature fitting capacity is enhanced, enabling the model to learn more complex class decision boundaries.

In the fine-tuning stage, the best model weights from the feature extraction stage are loaded, and several convolutional layers at the end of the backbone network become active. For convolutional layers located relatively in the middle (corresponding to general features), a small learning rate is applied. For layers closer to the output (corresponding to task-specific features), a moderate learning rate is applied. For the classification head (randomly initialized), a large learning rate is used to enable rapid convergence.

The two-stage transfer learning approach aims for:
- Preventing catastrophic forgetting of general features and fully leveraging pre-trained model knowledge. Freezing the entire backbone network during the feature extraction stage prevents the backpropagation process from overwriting the general features embedded in the pre-trained weights. Activating the later part of the backbone during the fine-tuning stage enables the model to better adapt to task-specific requirements of recognizing morphological characteristics of particular cat breeds.
- Suppressing overfitting. A variety of data augmentation techniques are applied at the data level (affine transformations, perspective distortion, rotation, flipping, blurring, local erasing, etc.), while stochastic dropout is employed at the model level to prevent neuron over-adaptation.
