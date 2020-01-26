package com.github.darrmirr.digitrecognizer.model.trainer;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.IOException;

public interface ModelTrainer {

    void train() throws IOException;

    boolean isNeedTrain();

    MultiLayerNetwork getTrainedModel() throws IOException;
}
