package com.github.darrmirr.digitrecognizer.model.trainer;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import static com.github.darrmirr.digitrecognizer.config.ApplicationConfiguration.PRE_TRAINED_MODEL_PATH;

@Component
@Qualifier("ConvolutionNNModelTrainer")
@Slf4j
public class ConvolutionNNModelTrainer implements ModelTrainer {
    private static final Path preTrainedModelPath = PRE_TRAINED_MODEL_PATH.resolve("bestModel.bin");
    @Autowired private EarlyStoppingTrainer trainer;

    @Override
    public void train() {
        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

        log.info("Termination reason: " + result.getTerminationReason());
        log.info("Termination details: " + result.getTerminationDetails());
        log.info("Total epochs: " + result.getTotalEpochs());
        log.info("Best epoch number: " + result.getBestModelEpoch());
        log.info("Score at best epoch: " + result.getBestModelScore());
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
