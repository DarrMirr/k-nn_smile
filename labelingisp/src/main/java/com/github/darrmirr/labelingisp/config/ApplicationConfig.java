package com.github.darrmirr.labelingisp.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import smile.data.parser.DelimitedTextParser;

@Configuration
public class ApplicationConfig {

    @Bean
    public DelimitedTextParser delimitedTextParser() {
        DelimitedTextParser parser = new DelimitedTextParser();
        parser.setDelimiter(",");
        return parser;
    }
}
