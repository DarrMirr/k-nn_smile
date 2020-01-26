package com.github.darrmirr.labelingisp.utils;

import javax.swing.*;
import java.awt.*;

public class PlotFrame extends JFrame {

    private PlotFrame() throws HeadlessException {
    }

    public static void print(JComponent jComponent) {
        var plotFrame = new PlotFrame();
        plotFrame.setContentPane(jComponent);
        plotFrame.setSize(400, 400);
        plotFrame.setLocationRelativeTo(null);
        plotFrame.setDefaultCloseOperation(EXIT_ON_CLOSE);
        plotFrame.setVisible(true);
    }

}
