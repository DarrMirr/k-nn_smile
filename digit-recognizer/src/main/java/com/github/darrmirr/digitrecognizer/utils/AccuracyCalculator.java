package com.github.darrmirr.digitrecognizer.utils;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
@Slf4j
public class AccuracyCalculator implements ScoreCalculator<MultiLayerNetwork> {
    private final DataSetIterator mnistTest;

    @Autowired
    public AccuracyCalculator(DataSetIterator mnistTest) {
        this.mnistTest = mnistTest;
    }

    private int i = 0;

    @Override
    public double calculateScore(MultiLayerNetwork network) {
        Evaluation evaluate = network.evaluate(mnistTest);
        double accuracy = evaluate.accuracy();
        log.info("Accuracy at iteration" + i++ + " " + accuracy);
        return 1 - evaluate.accuracy();
    }

    @Override
    public boolean minimizeScore() {
        return true;
    }
}
