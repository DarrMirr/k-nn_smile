package com.github.darrmirr.labelingisp.utils;

import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Component;

import java.util.function.Function;

@Component
public class Statistics {

    public Evaluation evaluate(int[] predictions, int[] labels, int numClasses) {
        var predictionVector = toVector().apply(predictions);
        var labelVector = toVector().apply(labels);

        var eval = new Evaluation(numClasses);
        eval.eval(labelVector, predictionVector);
        return eval;
    }

    public Function<int[], INDArray> toVector() {
        return array -> Nd4j
                .createFromArray(array)
                .reshape(array.length, 1);
    }
}
