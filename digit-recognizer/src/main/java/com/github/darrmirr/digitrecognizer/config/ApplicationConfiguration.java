package com.github.darrmirr.digitrecognizer.config;

import com.github.darrmirr.digitrecognizer.utils.AccuracyCalculator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

@Configuration
public class ApplicationConfiguration {
    /**
     * Number prediction classes.
     * We have 0-9 digits so 10 classes in total.
     */
    public static final int CLASS_COUNT = 10;

    /**
     * Mini batch gradient descent size or number of matrices processed in parallel.
     * For CORE-I7 16 is good for GPU please change to 128 and up
     */
    public static final int MINI_BATCH_SIZE = 16;// Number of training epochs

    /**
     * Number of total traverses through data.
     * with 5 epochs we will have 5/@MINI_BATCH_SIZE iterations or weights updates
     */
    public static final int EPOCHS = 5;

    /**
     * Number of total traverses through data. In this case it is used as the maximum epochs we allow
     * with 5 epochs we will have 5/@MINI_BATCH_SIZE iterations or weights updates
     *
     * used for training convolution neural network
     */
    private static final int MAX_EPOCHS = 20;

    /**
     * The alpha learning rate defining the size of step towards the minimum
     */
    public static final double LEARNING_RATE = 0.01;

    /**
     * https://en.wikipedia.org/wiki/Random_seed
     */
    public static final int SEED = 123;
    public static final int IMAGE_WIDTH = 28;
    public static final int IMAGE_HEIGHT = 28;
    public static final int CHANNELS = 1;
    public static final Path PRE_TRAINED_MODEL_PATH = Path.of("models", "training");


    @Bean
    public DataSetIterator mnistTrain() throws IOException {
        return new MnistDataSetIterator(MINI_BATCH_SIZE, true, SEED);
    }

    @Bean
    public DataSetIterator mnistTest() throws IOException {
        return new MnistDataSetIterator(MINI_BATCH_SIZE, false, SEED);
    }

    @Bean
    public ScoreIterationListener scoreIterationListener() {
        return new ScoreIterationListener(100);
    }

    @Bean
    public MultiLayerNetwork multiLayerNetwork(@Qualifier("simpleNN") MultiLayerConfiguration configuration, List<TrainingListener> trainingListenerList) {
        var model = new MultiLayerNetwork(configuration);
        trainingListenerList.forEach(model::addListeners);
        return model;
    }

    @Bean
    public EarlyStoppingConfiguration<MultiLayerNetwork> earlyStoppingConfiguration(AccuracyCalculator accuracyCalculator) {
        return new EarlyStoppingConfiguration
                .Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(MAX_EPOCHS))
                .scoreCalculator(accuracyCalculator)
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver(PRE_TRAINED_MODEL_PATH.toString()))
                .build();
    }

    @Bean
    public EarlyStoppingTrainer earlyStoppingTrainer(
            EarlyStoppingConfiguration<MultiLayerNetwork> earlyStoppingConfiguration,
            @Qualifier("convolutionNN") MultiLayerConfiguration configuration,
            DataSetIterator mnistTrain) {
        return new EarlyStoppingTrainer(earlyStoppingConfiguration, configuration, mnistTrain);
    }

    //NESTEROVS is referring to gradient descent with momentum
    @Bean
    public Nesterovs nesterovs(){
        return new Nesterovs.Builder()
                .learningRate(LEARNING_RATE)
                .build();
    }

    @Bean
    public ImagePreProcessingScaler imagePreProcessingScaler() {
        // Scale pixel values to 0-1
        return new ImagePreProcessingScaler(0, 1);
    }

    @Bean
    public ParentPathLabelGenerator parentPathLabelGenerator() {
        return new ParentPathLabelGenerator();
    }

    @Bean
    public ImageRecordReader imageRecordReader(PathLabelGenerator pathLabelGenerator) {
        return new ImageRecordReader(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, pathLabelGenerator);
    }
}
