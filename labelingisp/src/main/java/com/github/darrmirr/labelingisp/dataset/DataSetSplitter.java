package com.github.darrmirr.labelingisp.dataset;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.springframework.stereotype.Component;

import java.util.function.Function;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

@Component
public class DataSetSplitter {

    public SplitTestAndTrain split(DataSet dataSet, double fractionTrain) {
        int numTrain = (int) (fractionTrain * dataSet.numExamples());
        if (numTrain <= 0) {
            numTrain = 1;
        }
        DataSet train = getTrain(dataSet).apply(numTrain);
        DataSet test = getTest(dataSet).apply(numTrain);
        return new SplitTestAndTrain(train, test);
    }

    public SplitTestAndTrain shuffleAndSplit(DataSet dataSet, double fractionTrain) {
        dataSet.shuffle();
        return split(dataSet, fractionTrain);
    }

    public Function<Integer, DataSet> getTrain(DataSet dataSet) {
        return endIndex -> {
            DataSet train = new DataSet();
            train.setFeatures(dataSet.getFeatures().get(interval(0, endIndex), all()));
            train.setLabels(dataSet.getLabels().get(interval(0, endIndex)));
            return train;
        };
    }

    public Function<Integer, DataSet> getTest(DataSet dataSet) {
        return beginIndex -> {
            DataSet test = new DataSet();
            test.setFeatures(dataSet.getFeatures().get(interval(beginIndex, dataSet.numExamples()), all()));
            test.setLabels(dataSet.getLabels().get(interval(beginIndex, dataSet.numExamples())));
            return test;
        };
    }
}
