package com.github.darrmirr.labelingisp.dataset;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;
import smile.data.parser.DelimitedTextParser;

import java.io.IOException;
import java.io.InputStream;
import java.text.ParseException;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

@Component
public class DataSetReader {
    private static final Logger logger = LoggerFactory.getLogger(DataSetReader.class);
    private DelimitedTextParser parser;

    @Autowired
    public DataSetReader(DelimitedTextParser parser) {
        this.parser = parser;
    }

    public DataSet readCSV(Resource resource) {
        try(InputStream inputStream = resource.getInputStream()){
            var datasetParsed = parser.parse(resource.getFilename(), inputStream);
            var datasetMatrix = Nd4j.createFromArray(datasetParsed.x());
            return new DataSet(datasetMatrix.get(all(), interval(0, 2)), datasetMatrix.get(all(), point(2)));
        } catch (IOException | ParseException e) {
            logger.error("error to read csv resource " + resource, e);
        }
        return DataSet.empty();
    }
}
