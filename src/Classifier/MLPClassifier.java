/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Classifier;

import java.util.Random;
import java.util.Vector;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author USER
 */
public class MLPClassifier extends AbstractClassifier {    
    
    /**
     * A single perceptron container/helper class
     */
    private class Perceptron {
	
	/**
	 * The value and weight of all of this perceptron input 
	 * w0 is in index 0 and always had the value of 1
	 */
	private final Vector<Double> InputValue = new Vector<>();
	private final Vector<Double> InputWeight = new Vector<>();
	
	Random rand = new Random();
	
	
	/**
	 * Intiate the perceptron, add x0 with value 1 and a random weight
	 */
	public Perceptron() {
	    rand.setSeed(System.currentTimeMillis());
	    InputValue.add(1.0);
	    InputWeight.add(rand.nextDouble());
	}
	
	/**
	 * Add another input with a random weight, return the new input weight
	 */
	public int addInput(double value) {
	    InputValue.add(value);
	    InputWeight.add(rand.nextDouble());
	    return InputValue.size()-1;
	}
	
	/**
	 * Set the value of this input to the paramater's
	 */
	public void setValue(int index, double value) {
	    InputValue.set(index, value);
	}
	
	/**
	 * Return the sum of all input's value times weight
	 */
	public double getRawOutput() {
	    double res = 0;
	    for(int i=0; i<InputValue.size(); i++) {
		res+= InputValue.get(i) * InputWeight.get(i);
	    }
	    return res;
	}
	
	/**
	 * Return the output with a sigmoid activation function
	 */
	public double getOutput() {
	    return 1/(1+Math.exp(-(getRawOutput())));
	}
	
	/**
	 * Improve each of the InputWeight entry according to the error value
	 * ERROR MUST target-actual OR SUM OF ERROR OF FRONT PERCEPTRON
	 * WHICH THE ERROR IS COUNTED BY target-actual
	 */
	public void learn(double rate, double error) {
	    for(int i=0; i<InputWeight.size(); i++) {
		double newWeight = InputWeight.get(i) + rate*error*InputValue.get(i);
		InputWeight.set(i, newWeight);
	    }
	}
	
    }
    
    /**
     * A multi-layer perceptron container/helper class
     */
    private class MultiLayerPerceptron {
	
    }
    
    /**
     * Generates a classifier. Must initialize all fields of the classifier that 
     * are not being set via options (ie. multiple calls of buildClassifier must 
     * always lead to the same result). Must not change the dataset in any way.
     *
     * @param data set of instances serving as training data
     * @exception Exception if the classifier has not been generated successfully
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
    }
    
    /**
     * Predicts the class memberships for a given instance. If an instance is 
     * unclassified, the returned array elements must be all zero. If the class 
     * is numeric, the array must consist of only one element, which contains 
     * the predicted value. Note that a classifier MUST implement either this 
     * or classifyInstance().
     *
     * @param instance the instance to be classified
     * 
     * @return an array containing the estimated membership probabilities of the 
     * test instance in each class or the numeric prediction
     * 
     * @exception Exception if distribution could not be computed successfully
     */
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
	return null;
    }
    
}
