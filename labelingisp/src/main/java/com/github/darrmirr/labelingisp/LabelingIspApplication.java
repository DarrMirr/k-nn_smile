package com.github.darrmirr.labelingisp;

import com.github.darrmirr.labelingisp.dataset.DataSetReader;
import com.github.darrmirr.labelingisp.dataset.DataSetSplitter;
import com.github.darrmirr.labelingisp.utils.PlotFrame;
import com.github.darrmirr.labelingisp.utils.Statistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.core.io.Resource;
import smile.classification.KNN;
import smile.plot.ScatterPlot;

import java.awt.*;

/**
 * Java implementation of K-NN application
 * from article http://xyclade.github.io/MachineLearning/#labeling-isps-based-on-their-downupload-speed-k-nn-using-smile-in-scala
 */

@SpringBootApplication
public class LabelingIspApplication {
	private static final Logger logger = LoggerFactory.getLogger(LabelingIspApplication.class);
	private DataSetReader dataSetReader;
	private DataSetSplitter dataSetSplitter;
	private Statistics statistics;
	private Resource resource;

	@Autowired
	public LabelingIspApplication(
			DataSetReader dataSetReader,
			DataSetSplitter dataSetSplitter,
			Statistics statistics,
			@Value("classpath:data/KNN_Example_1.csv") Resource resource
	) {
		this.dataSetReader = dataSetReader;
		this.dataSetSplitter = dataSetSplitter;
		this.statistics = statistics;
		this.resource = resource;
	}

	public static void main(String[] args) {
		new SpringApplicationBuilder(LabelingIspApplication.class)
				.headless(false)
				.run(args);
	}

	@EventListener
	public void startApp(ApplicationReadyEvent event) {
		var dataSet = dataSetReader.readCSV(resource);

		var plotCanvas = ScatterPlot.plot(dataSet.getFeatures().toDoubleMatrix(), dataSet.getLabels().toIntVector(), '@', new Color[]{ Color.RED, Color.BLUE });
		PlotFrame.print(plotCanvas);

		var splitTestAndTrain = dataSetSplitter.shuffleAndSplit(dataSet, 0.80D);

		var trainDataSet = splitTestAndTrain.getTrain();
		var cvDataSet = splitTestAndTrain.getTest();
		var knn = KNN.learn(trainDataSet.getFeatures().toDoubleMatrix(), trainDataSet.getLabels().toIntVector(), 3);
		var cvPredictionArray = knn.predict(cvDataSet.getFeatures().toDoubleMatrix());

		var evaluation = statistics.evaluate(cvPredictionArray, cvDataSet.getLabels().toIntVector(), 2);
		logger.info(evaluation.stats());

		var result = knn.predict(new double[]{ 5.3D, 4.3D });
		logger.info("prediction result : {}", result);
	}
}
