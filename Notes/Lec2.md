neural network
- input & output fixed (what you have / want)
- hidden layer alone contains variability through **weights of the inputs**
- dense linked, no trespassing
- each layer as a whole act to detect certain features
- mathematical essence: complex function

activation functions: linear for regression, sigmoid for binary classification, softmax for multi-class classification

parameter / hyperparameter
- parameter changes during the training process
- hyperparameter is determined by human in advance

---

simple neural network (with flattened data as input) doesn't contain sequential information

image recognition features
- translational invariance: the identification of a certain figure is independent to its location
- locality: pixels nearby are highly correlated
- hence, earlier layers should focus on local features, and deeper layers should focus on long-range features

convolution
- using the same kernel to ensure translational invariance
- the kernel only make dot multiplication with the same size of rectagular pixels without engaging other pixels
- each kernel can only detect one kind of feature; introducing multipule kernels to simultaneously extract different features (forming feature maps, each feature as a channel)

CNN: learn the kernels as parameters

padding
- problem: less information extracted from the perimeter of the image
- solution: increase the size of image by adding extra "0" pixels outside the perimeter

striding: increase the step size to downsample & increase computational efficiency

for colored images
- each kernel does tensor calculation (channels matched with the color channels each)
- (for each kernel) still get **a single channel as output** by adding results from channels up

pooling: mitigate the sensitivity of convolutional layer to location & downsample
- type: max-pooling & average pooling
