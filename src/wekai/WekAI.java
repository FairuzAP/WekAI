/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekai;

import weka.classifiers.trees.J48;
import Classifier.Tree.ID3Classifier;
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
	    w.readData("C:/Program Files/Weka-3-8/data/vote.arff");
	    ID3Classifier id3 = new ID3Classifier();
	    J48 j = new J48();
	    
	    w.Model = id3;
	    w.fullTraining();

	} catch (Exception ex) {
	    Logger.getLogger(WekAI.class.getName()).log(Level.SEVERE, null, ex);
	}
	
    }
    
}

/*
            Scanner input = new Scanner(System.in);

	    System.out.println("Classifier yang akan digunakan?");
            System.out.println("1. NaiveBayes");
            System.out.println("2. FFNN");
            System.out.println("3. Baca model faiz.model");
            int model = input.nextInt();
            if (model == 1) {
		w.readData("D:/iris.arff");
                w.Model = nb;
            } else if (model == 2) {
                w.readData("D:/Team.arff");
                w.Model = n;
                System.out.println("Input perceptron number in hidden layer (0 for no hidden layer) : ");
                int models = input.nextInt();
                if (models > 0) {
                    temp.add(models);
                }
                n.setPerceptronCount(temp);
            } else if (model == 3) {
                w.readData("D:/iris.arff");
		w.readModel("D:/faiz.model");
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
		w.randomize();
                w.tenFoldCrossValidation();
            } else {
                System.out.println("input salah");
                System.exit(0);
            }
	    w.saveModel("D:/Team-test.model");
*/
	    //Train and save model student-train NB
//	    w.readData("D:/student-train.arff");
//	    w.Model = nb;
//	    w.fullTraining();
//	    w.saveModel("D:/std.model");
//	    
	    //Student-mat-test FFNN
//	    w.readData("D:/student-mat-test.arff");
//	    w.readModel("D:/std2.model");
//	    w.fullTrainingTest();

	    //Student-mat-test NB
//	    w.readData("D:/student-mat-test.arff");
//	    w.readModel("D:/std.model");
//	    w.fullTrainingTest();