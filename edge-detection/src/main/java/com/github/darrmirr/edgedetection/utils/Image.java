package com.github.darrmirr.edgedetection.utils;

import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Objects;

import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class Image {
    @Getter private final INDArray image;

    public Image(INDArray image) {
        Objects.requireNonNull(image);
        if (image.rank() != 4) {
            throw new IllegalStateException("image should have rank equal to 4");
        }
        this.image = image;
    }

    public int getWidth() {
        return (int) image.size(3);
    }

    public int getHeight() {
        return (int) image.size(2);
    }

    public double[][] toRedChannel() {
        return image.get(point(0), point(0)).toDoubleMatrix();
    }

    public double[][] toGreenChannel() {
        return image.get(point(0), point(1)).toDoubleMatrix();
    }

    public double[][] toBlueChannel() {
        return image.get(point(0), point(2)).toDoubleMatrix();
    }
}
