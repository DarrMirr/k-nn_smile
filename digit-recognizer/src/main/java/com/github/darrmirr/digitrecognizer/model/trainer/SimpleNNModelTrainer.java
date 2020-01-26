package com.github.darrmirr.digitrecognizer.model.trainer;

import com.github.darrmirr.digitrecognizer.config.ApplicationConfiguration;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import static com.github.darrmirr.digitrecognizer.config.ApplicationConfiguration.PRE_TRAINED_MODEL_PATH;

@Component
@Qualifier("SimpleNNModelTrainer")
@Slf4j
public class SimpleNNModelTrainer implements ModelTrainer {
    private static final Path preTrainedModelPath = PRE_TRAINED_MODEL_PATH.resolve("simplePreTrainedModelMnist.zip");
    @Autowired private DataSetIterator mnistTrain;
    @Autowired private DataSetIterator mnistTest;
    @Autowired private MultiLayerNetwork model;

    @Override
    public void train() throws IOException {
        preTrain();
        saveModel();
    }

    private void preTrain() {
        log.info("Start train model");
        for (int i = 1; i < ApplicationConfiguration.EPOCHS ; i++) {
            model.fit(mnistTrain);
            log.info("*** Completed epoch : {}***", i);

            log.info("Evaluate model....");
            var evaluation = model.evaluate(mnistTest);

            log.info(evaluation.stats());
            if (evaluation.accuracy() >= 0.97) {
                log.info("Congratulations,the desired score found,!");
                break;
            }
            mnistTest.reset();
        }
        log.info("Finish train model");
    }

    private void saveModel() throws IOException {
        log.info("Saving model : {}", preTrainedModelPath.toAbsolutePath());
        try(OutputStream outputStream = Files.newOutputStream(preTrainedModelPath)) {
            ModelSerializer.writeModel(model, outputStream, true);
        }
    }

    @Override
    public boolean isNeedTrain() {
        return !preTrainedModelPath.toFile().exists();
    }

    @Override
    public MultiLayerNetwork getTrainedModel() throws IOException {
        log.info("Restoring model : {}", preTrainedModelPath.toAbsolutePath());
        try(InputStream inputStream = Files.newInputStream(preTrainedModelPath)) {
            return ModelSerializer.restoreMultiLayerNetwork(inputStream);
        }
    }
}
