package com.github.darrmirr.digitrecognizer.config;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import static com.github.darrmirr.digitrecognizer.config.ApplicationConfiguration.*;

/*
 * Notice:
 *   It is need to re-train NN in case of configuration is changed
 *
 */

@Configuration
public class SimpleNeuralNetworkConfiguration {

    @Bean
    @Qualifier("simpleNN")
    public MultiLayerConfiguration neuralNetConfiguration(Nesterovs nesterovs) {
        return new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .weightInit(WeightInit.XAVIER)
                .updater(nesterovs)
                .list()
                    .layer(0, inputLayer())
                    .layer(1, firstHiddenLayer())
                    .layer(2, outputLayer())
                .setInputType(InputType.convolutional(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)) // see comments ConvolutionNeuralNetworkConfiguration.java
                .build();
    }

    /**
     * First layer (input layer) uses 128 neurons each with RELU activation.
     * Is called dense layer because every neuron is linked with
     * every other neuron on next layer and previous layer
     */
    @Bean
    public Layer inputLayer() {
        return new DenseLayer.Builder()
                .activation(Activation.RELU)
                .nIn(IMAGE_WIDTH * IMAGE_HEIGHT)
                .nOut(128)
                .build();
    }


    /**
     * First hidden layer uses 64 neurons each with RELU activation.
     */
    @Bean
    public Layer firstHiddenLayer() {
        return new DenseLayer.Builder()
                .activation(Activation.RELU)
                .nOut(64)
                .build();
    }

    /**
     * The output layer using SOFTMAX to predict 10 classes(CLASS_COUNT)
     * NEGATIVELOGLIKELIHOOD is just a cost function measuring how
     * good our prediction or hypothesis is doing against real digits
     */
    @Bean
    public Layer outputLayer() {
        return new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nOut(CLASS_COUNT)
                .build();
    }
}
