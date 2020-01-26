package com.github.darrmirr.edgedetection.utils;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Paths;

@Component
@AllArgsConstructor
@Slf4j
public class ImageProcessor {
    @Autowired private NativeImageLoader imageLoader;

    /**
     * Transform image to matrix
     *
     * @param resource contain image resource is loaded form disk, incoming request or another source
     * @return matrix of resource
     * @throws IOException
     */
    public INDArray asMatrix(Resource resource) {
        try(InputStream io = resource.getInputStream()) {
            return imageLoader.asMatrix(io);
        }  catch (IOException e) {
            throw new IllegalStateException("error to transform image " + resource.getFilename() + " as matrix", e);
        }
    }

    /**
     * Transform image to matrix and reshape it in order to make fit for model
     *
     * @param resource contain image resource is loaded form disk, incoming request or another source
     * @param matrixDimension contain new shape (matrix dimension) for matrix.
     * @return matrix of resource
     * @throws IllegalStateException
     */
    public INDArray asMatrix(Resource resource, int[] matrixDimension) {
        return asMatrix(resource)
                .reshape(matrixDimension);
    }

    /**
     * Wrap INDArray to Image object
     *
     * @param resource contain image resource is loaded form disk, incoming request or another source
     * @return wrapper object called as image
     */
    public Image asImage(Resource resource) {
        return new Image(asMatrix(resource));
    }

    /**
     *
     * @param imageRGB image matrix
     * @param imageWidth image width
     * @param imageHeight image height
     * @param fileName image filename
     * @throws IOException
     */
    public void save(double[][] imageRGB, int imageWidth, int imageHeight, String fileName) throws IOException {
        BufferedImage writeBackImage = new BufferedImage(imageWidth, imageHeight, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < imageRGB.length; i++) {
            for (int j = 0; j < imageRGB[i].length; j++) {
                Color color = new Color((int)imageRGB[i][j], (int)imageRGB[i][j], (int)imageRGB[i][j]);
                writeBackImage.setRGB(j, i, color.getRGB());
            }
        }
        ImageIO.write(writeBackImage, "png", Paths.get(fileName).toFile());
    }
}
