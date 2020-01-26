package com.github.darrmirr.digitrecognizer.utils;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.Random;

import static com.github.darrmirr.digitrecognizer.config.ApplicationConfiguration.CLASS_COUNT;
import static com.github.darrmirr.digitrecognizer.config.ApplicationConfiguration.SEED;

@Component
public class MnistImageLoader {
    private ImageRecordReader imageRecordReader;
    private DataNormalization dataNormalization;
    private Resource testDigitDir;
    private Random randNumGen = new Random(SEED);

    @Autowired
    public MnistImageLoader(
            ImageRecordReader imageRecordReader,
            DataNormalization dataNormalization,
            @Value("classpath:test_digit") Resource testDigitDir
    ) {
        this.imageRecordReader = imageRecordReader;
        this.dataNormalization = dataNormalization;
        this.testDigitDir = testDigitDir;
    }

    public DataSetIterator getTestDataSet() throws IOException {
        FileSplit test = new FileSplit(testDigitDir.getFile(), NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        imageRecordReader.initialize(test);

        // Test the Loaded Model with the test data
        DataSetIterator testIter = new RecordReaderDataSetIterator(imageRecordReader, 128, 1, CLASS_COUNT);

        // Scale pixel values to 0-1
        dataNormalization.fit(testIter);
        testIter.setPreProcessor(dataNormalization);
        return testIter;
    }


}
