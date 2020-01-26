package com.github.darrmirr.digitrecognizer.config;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Scope;

import static com.github.darrmirr.digitrecognizer.config.ApplicationConfiguration.*;


@Configuration
public class ConvolutionNeuralNetworkConfiguration {

    /*
     * Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
     * (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
     * and the dense layer
     * (b) Does some additional configuration validation
     * (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
     * layer based on the size of the previous layer (but it won't override values manually set by the user)
     * InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
     * For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
     * MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
     * row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
     *
     * source: https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/LenetMnistExample.java#L114
     */
    @Bean
    @Qualifier("convolutionNN")
    public MultiLayerConfiguration convolutionNeuralNetConfiguration(Nesterovs nesterovs) {
        return new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .weightInit(WeightInit.XAVIER)
                .updater(nesterovs)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                    .layer(0, firstConvolutionLayer())
                    .layer(1, maxPoolingLayer())
                    .layer(2, secondConvolutionLayer())
                    .layer(3, maxPoolingLayer())
                    .layer(4, firstHiddenLayerCNN())
                    .layer(5, secondHiddenLayerCNN())
                    .layer(6, outputLayerCNN())
                .setInputType(InputType.convolutionalFlat(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)) // see comments above
                .build();
    }

    @Bean
    public Layer firstConvolutionLayer() {
        return new ConvolutionLayer.Builder()
                .nIn(CHANNELS)
                .nOut(20)
                .kernelSize(5, 5)
                .stride(1, 1)
                .activation(Activation.IDENTITY)
                .build();
    }

    @Bean
    @Scope("prototype")
    public Layer maxPoolingLayer() {
        return new SubsamplingLayer.Builder()
                .poolingType(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build();
    }

    @Bean
    public Layer secondConvolutionLayer() {
        return new ConvolutionLayer.Builder()
                .nIn(20)
                .nOut(50)
                .kernelSize(5, 5)
                .stride(1, 1)
                .activation(Activation.IDENTITY)
                .build();
    }

    @Bean
    public Layer firstHiddenLayerCNN() {
        return new DenseLayer.Builder()
                .activation(Activation.RELU)
                .nIn(800)
                .nOut(128)
                .build();
    }

    @Bean
    public Layer secondHiddenLayerCNN() {
        return new DenseLayer.Builder()
                .activation(Activation.RELU)
                .nIn(128)
                .nOut(64)
                .build();
    }

    @Bean
    public Layer outputLayerCNN() {
        return new OutputLayer.Builder()
                .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nOut(CLASS_COUNT)
                .build();
    }
}
