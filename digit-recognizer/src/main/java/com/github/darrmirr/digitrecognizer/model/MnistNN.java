package com.github.darrmirr.digitrecognizer.model;

import com.github.darrmirr.digitrecognizer.model.trainer.ModelTrainer;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.io.IOException;

@Component
@Slf4j
@AllArgsConstructor
@NoArgsConstructor
public class MnistNN {
    @Autowired
//    @Qualifier("SimpleNNModelTrainer")
    @Qualifier("ConvolutionNNModelTrainer")
    private ModelTrainer modelTrainer;
    private MultiLayerNetwork model;

    @PostConstruct
    public void init() throws IOException {
        if (modelTrainer.isNeedTrain()) {
            modelTrainer.train();
        }
        model = modelTrainer.getTrainedModel();
    }

    public int predict(double[] imageVector) {
        return predict(Nd4j.create(new double[][]{imageVector}));
    }

    public int  predict(double[][] imageMatrix) {
        return predict(Nd4j.create(imageMatrix));
    }

    public int predict(INDArray image) {
        int[] predict = model.predict(image);
        return predict[0];
    }

    public INDArray output(DataSet dataSet) {
        return model.output(dataSet.getFeatures());
    }
}
