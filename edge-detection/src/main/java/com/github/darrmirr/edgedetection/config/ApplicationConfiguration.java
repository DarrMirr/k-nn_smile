package com.github.darrmirr.edgedetection.config;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.HashMap;
import java.util.Map;

@Configuration
public class ApplicationConfiguration {
    public static final String HORIZONTAL_FILTER = "Horizontal Filter";
    public static final String VERTICAL_FILTER = "Vertical Filter";

    public static final String SOBEL_FILTER_VERTICAL = "Sobel Vertical Filter";
    public static final String SOBEL_FILTER_HORIZONTAL = "Sobel Horizontal Filter";

    public static final String SCHARR_FILTER_VETICAL = "Scharr Vertical Filter";
    public static final String SCHARR_FILTER_HORIZONTAL = "Scharr Horizontal Filter";

    private static final double[][] FILTER_VERTICAL = {{1, 0, -1}, {1, 0, -1}, {1, 0, -1}};
    private static final double[][] FILTER_HORIZONTAL = {{1, 1, 1}, {0, 0, 0}, {-1, -1, -1}};

    private static final double[][] FILTER_SOBEL_V = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    private static final double[][] FILTER_SOBEL_H = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    private static final double[][] FILTER_SCHARR_V = {{3, 0, -3}, {10, 0, -10}, {3, 0, -3}};
    private static final double[][] FILTER_SCHARR_H = {{3, 10, 3}, {0, 0, 0}, {-3, -10, -3}};

    @Bean
    public Map<String, INDArray> filterMap() {
        Map<String, INDArray> filterMap;
        filterMap = new HashMap<>();
        filterMap.put(VERTICAL_FILTER, Nd4j.createFromArray(FILTER_VERTICAL));
        filterMap.put(HORIZONTAL_FILTER, Nd4j.createFromArray(FILTER_HORIZONTAL));

        filterMap.put(SOBEL_FILTER_VERTICAL, Nd4j.createFromArray(FILTER_SOBEL_V));
        filterMap.put(SOBEL_FILTER_HORIZONTAL, Nd4j.createFromArray(FILTER_SOBEL_H));

        filterMap.put(SCHARR_FILTER_VETICAL, Nd4j.createFromArray(FILTER_SCHARR_V));
        filterMap.put(SCHARR_FILTER_HORIZONTAL, Nd4j.createFromArray(FILTER_SCHARR_H));
        return filterMap;
    }

    @Bean
    public NativeImageLoader nativeImageLoader() {
        return new NativeImageLoader();
    }
}
