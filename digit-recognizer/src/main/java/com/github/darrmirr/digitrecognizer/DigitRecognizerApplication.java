package com.github.darrmirr.digitrecognizer;

import com.github.darrmirr.digitrecognizer.model.MnistNN;
import com.github.darrmirr.digitrecognizer.utils.MnistImageLoader;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.event.ApplicationStartedEvent;
import org.springframework.context.event.EventListener;

import java.io.IOException;

import static com.github.darrmirr.digitrecognizer.config.ApplicationConfiguration.CLASS_COUNT;

/*
 * Picture's requirement:
 *   - background - black color
 *   - foreground (digit) - white color
 */

@Slf4j
@SpringBootApplication
public class DigitRecognizerApplication {
	@Autowired
	private MnistNN mnistNN;
	@Autowired
	private MnistImageLoader mnistImageLoader;

	public static void main(String[] args) {
		SpringApplication.run(DigitRecognizerApplication.class, args);
	}

	@EventListener
	public void startApp(ApplicationStartedEvent event) throws IOException {
		var testDataSet = mnistImageLoader.getTestDataSet();

		// Create Eval object with 10 possible classes
		Evaluation eval = new Evaluation(CLASS_COUNT);

		while (testDataSet.hasNext()) {
			DataSet next = testDataSet.next();
			INDArray output = mnistNN.output(next);
			eval.eval(next.getLabels(), output);
		}

		log.info(eval.stats());
	}
}
