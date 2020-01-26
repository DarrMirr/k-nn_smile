package com.github.darrmirr.edgedetection;

import com.github.darrmirr.edgedetection.utils.Convolution;
import com.github.darrmirr.edgedetection.utils.ImageProcessor;
import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.event.ApplicationStartedEvent;
import org.springframework.context.event.EventListener;
import org.springframework.core.io.Resource;

import java.io.IOException;
import java.util.Map;

import static com.github.darrmirr.edgedetection.config.ApplicationConfiguration.VERTICAL_FILTER;

@SpringBootApplication
@AllArgsConstructor
@NoArgsConstructor
public class EdgeDetectionApplication {
	@Autowired private Map<String, INDArray> filterMap;
	@Autowired private ImageProcessor imageProcessor;
	@Autowired private Convolution convolution;
	@Value("classpath:pictures/butterfly.jpg")
	private Resource imageResource;

	public static void main(String[] args) {
		SpringApplication.run(EdgeDetectionApplication.class, args);
	}

	@EventListener
	public void startApp(ApplicationStartedEvent event) throws IOException {
		var image = imageProcessor.asImage(imageResource);
		var convolvedPixels = convolution.apply(image, filterMap.get(VERTICAL_FILTER).toDoubleMatrix());
		imageProcessor.save(convolvedPixels, image.getWidth(), image.getHeight(), "result.png");
	}
}
