/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekai;

import Classifier.NaiveBayesClassifier;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author USER
 */
public class WekAI {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
	
	try {
	    
	    WekaHandler w = new WekaHandler();
	    w.readData("C:/Program Files/Weka-3-8/data/iris.arff");
	    NaiveBayesClassifier n = new NaiveBayesClassifier();
	    w.Model = n;
	    w.tenFoldCrossValidation();
	    
	} catch (Exception ex) {
	    Logger.getLogger(WekAI.class.getName()).log(Level.SEVERE, null, ex);
	}
	
    }
    
}
