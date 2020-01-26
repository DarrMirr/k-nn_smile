# Handwritten digit recognizer application

**Requirement:**
- java 11
- maven 3

**Project structure:**
- config (application configuration)
- model (classes working with model)
- utils (useful utility classes)
- resources (store test images)

**Description:**

There is two neural network in application which used to recognize handwritten digit.
First one is called Simple Neural Network. Second one is called Convolution Neural Network.
Both are used [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) to train. MNIST dataset is downloaded automatically by MnistFetcher class.
Download process is performed while MnistDataSetIterator class is created.

Train process takes time to train. Simple Neural Network is trained during several minutes.
Convolution one is trained during 30-60 minutes. Both trained models are stored into `ApplicationConfiguration.PRE_TRAINED_MODEL_PATH`.
Train process is not perform in case of trained model exists.

Comment `@Qualifier("SimpleNNModelTrainer")`  and uncomment `@Qualifier("ConvolutionNNModelTrainer")` lines in MnistNN class in order to switch model to use and vice verse.

Run main() method in DigitRecognizerApplication class in order to recognize handwritten digit which stored in resources directory.
Feel free to add new images to `resources/test_digit directory`. Application automatically read all image files from subdirectories called from 0 to 9.

There is requirement to images:
- size: 28x28 pixels.
- handwritten digit color: white
- background color: black

I have tried to invert image colors. Both model have very poor prediction accuracy in this case.

Examples result output:

**Simple Neural Network**
```
========================Evaluation Metrics========================
 # of classes:    10
 Accuracy:        0,6190
 Precision:       0,6296	(1 class excluded from average)
 Recall:          0,6333
 F1 Score:        0,6444	(1 class excluded from average)
Precision, recall & F1: macro-averaged (equally weighted avg. of 10 classes)

Warning: 1 class was never predicted by the model and was excluded from average precision
Classes excluded from average precision: [8]

=========================Confusion Matrix=========================
 0 1 2 3 4 5 6 7 8 9
---------------------
 2 0 0 0 0 0 0 0 0 0 | 0 = 0
 0 0 0 0 0 0 1 0 0 1 | 1 = 1
 0 0 2 0 0 0 0 0 0 0 | 2 = 2
 0 1 0 1 0 0 0 0 0 0 | 3 = 3
 0 0 0 0 2 0 0 0 0 0 | 4 = 4
 0 0 0 0 0 2 0 0 0 0 | 5 = 5
 0 0 0 0 0 0 2 0 0 0 | 6 = 6
 0 0 1 0 0 0 0 1 0 0 | 7 = 7
 0 0 0 1 0 0 0 0 0 1 | 8 = 8
 0 0 0 1 0 1 0 0 0 1 | 9 = 9
```

**Convolution Neural Network**
```
========================Evaluation Metrics========================
 # of classes:    10
 Accuracy:        0,8095
 Precision:       0,8900
 Recall:          0,8333
 F1 Score:        0,8305
Precision, recall & F1: macro-averaged (equally weighted avg. of 10 classes)


=========================Confusion Matrix=========================
 0 1 2 3 4 5 6 7 8 9
---------------------
 2 0 0 0 0 0 0 0 0 0 | 0 = 0
 0 2 0 0 0 0 0 0 0 0 | 1 = 1
 0 0 2 0 0 0 0 0 0 0 | 2 = 2
 0 0 1 1 0 0 0 0 0 0 | 3 = 3
 0 0 0 0 2 0 0 0 0 0 | 4 = 4
 0 0 0 0 0 2 0 0 0 0 | 5 = 5
 0 0 0 0 0 0 2 0 0 0 | 6 = 6
 0 0 0 0 0 0 0 2 0 0 | 7 = 7
 0 0 0 0 0 0 0 0 1 1 | 8 = 8
 0 0 2 0 0 0 0 0 0 1 | 9 = 9

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
```

As expected Convolution Neural Network have better result than Simple one.

**How does it works**

Notice: it is high level description about how does it works.

Simple Neural Network have three layers: 
 - input
 - hidden
 - output 

Input and hidden are dense layers (fully connected layers). Input layer have 784 (28x28 pixels) units (neurons). All units have special weights. Weights are numbers that used for computation mathematics function inside unit.

Purpose of train process neural network is to calculate weights for mathematics function for each unit that all units in neural network make required result with maximum accuracy.

For Handwritten digit recognizer application:
- required result: predict one of this classes [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]. In reality, output of NN is vector that contains numbers from 0.0 (0%) to 1.0 (100%) for each class.  
- maximum accuracy: up to 100%

There is always error in computations. So, 100% accuracy have incredibly high computation cost.

Handwritten Image dataset is required in order to train neural network. MNIST dataset is used to train neural network.
Simple Neural Network have 97% accuracy on MNIST test data during training process. But accuracy is decrease to 61% when I use different images. (See Simple Neural Network result above)

One of the method to increase accuracy is using Convolution Neural Network.

Convolution Neural Network have additional layers. This layers are placed before Simple Neural Network Layers.
So, additional layers is used to make additional work on image in order to increase prediction accuracy.

There is 4 additional layers in Convolution Neural Network:

- ConvolutionLayer
- SubsamplingLayer (called MaxPoolingLayer)
- ConvolutionLayer
- SubsamplingLayer (called MaxPoolingLayer)

`ConvolutionLayer` is used to edge detection in digital image.

According to [wikipedia](https://en.wikipedia.org/wiki/Edge_detection):
>Edge detection includes a variety of mathematical methods that aim at identifying points in a digital image at which the image brightness changes sharply

Different image filters, such as [Sobel filter](https://en.wikipedia.org/wiki/Sobel_operator), are applying to input picture in Convolution Layer.
Image edge can be treat as feature of input picture. Convolution Layer significantly increase number of features.

`MaxPoolingLayer` is used to decrease number of features. Only strong features (having maximum number at window with size equal to [kernel x kernel]) is kept.

Training process is the same as Simple Neural Network.
Convolution Neural Network have 98% accuracy on MNIST test data during training process. But accuracy is more high than Simple Neural Network on the same test images. (80% vs. 61%)