/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekai;

import Classifier.FFNNClassifier;
import Classifier.NaiveBayesClassifier;
import java.util.Scanner;
import java.util.Vector;
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
	    //w.readData("D:/mush.arff");
	    //w.readData("C:/Program Files/Weka-3-8/data/iris.arff");
	    FFNNClassifier n = new FFNNClassifier();
            NaiveBayesClassifier nb = new NaiveBayesClassifier();
	    Vector<Integer> temp = new Vector<>();

            Scanner input = new Scanner(System.in);


            System.out.println("Classifier yang akan digunakan?");
            System.out.println("1. NaiveBayes");
            System.out.println("2. FFNN");
            int model = input.nextInt();
            if (model == 1) {
                w.readData("D:/ProjectNow/AI/Mush.arff");
                w.Data.setClassIndex(0);
                w.Model = nb;
            } else if (model == 2) {
                w.readData("D:/ProjectNow/AI/Team.arff");
                w.Model = n;
                System.out.println("Input perceptron number in hidden layer (0 for no hidden layer) : ");
                model = input.nextInt();
                if (model > 0) {
                    temp.add(model);
                    //temp.add(20);
                }
                n.setPerceptronCount(temp);
            } else {
                System.out.println("input salah");
                System.exit(0);
            }
            
            System.out.println("Metode training yang akan digunakan?");
            System.out.println("1. Full Training");
            System.out.println("2. 10 Fold Cross Validation");
            int inInt = input.nextInt();
            System.out.println("Processing...");
            if (inInt == 1) {
                if (model == 2) {
                    n.setMaxEpoch(2000);
                    n.setLearningRate(0.1);
                    n.setTarget(0);
                }
                w.fullTraining();
            } else if (inInt == 2) {
                if (model == 2) {
                    n.setMaxEpoch(1000);
                    n.setLearningRate(0.2);
                    n.setTarget(0);
                }
                w.tenFoldCrossValidation();
            } else {
                System.out.println("input salah");
                System.exit(0);
            }
	} catch (Exception ex) {
	    Logger.getLogger(WekAI.class.getName()).log(Level.SEVERE, null, ex);
	}
	
    }
    
}
