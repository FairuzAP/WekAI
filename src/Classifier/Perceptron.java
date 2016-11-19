/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Classifier;

import java.io.Serializable;
import java.util.Random;
import java.util.Vector;

/** A single perceptron container/helper class */
class Perceptron implements Serializable {
    
    /**
     * The value and weight of all of this perceptron input 
     * w0 is in index 0 and always had the value of 1
     */
    private final Vector<Double> InputValue = new Vector<>();
    private final Vector<Double> InputWeight = new Vector<>();

    static Random rand = new Random(System.currentTimeMillis());


    /** Intiate the perceptron, add x0 with value 1 and a random weight */
    public Perceptron() {
	InputValue.add(1.0);
	InputWeight.add(rand.nextDouble()-0.5);
    }


    /** Add another input with a random weight, return the new input weight */
    public int addInput(double value) {
	InputValue.add(value);
	InputWeight.add(rand.nextDouble()-0.5);
	return InputValue.size()-1;
    }

    /** Set the value of this input to the paramater's */
    public void setValue(int index, double value) {
	InputValue.set(index, value);
    }

    /** Return the sum of all input's value times weight */
    public double getRawOutput() {
	double res = 0;
	for(int i=0; i<InputValue.size(); i++) {
	    res+= InputValue.get(i) * InputWeight.get(i);
	}
	return res;
    }

    /** Return the output with a sigmoid activation function */
    public double getOutput() {
	return 1/(1+Math.exp(-(getRawOutput())));
    }

    public double getWeight(int index) {
	return InputWeight.get(index);
    }

    /** Improve each of the InputWeight entry according to the totError value */
    public void learn(double rate, double error) {
	for(int i=0; i<InputWeight.size(); i++) {
	    double newWeight = InputWeight.get(i) + rate*error*InputValue.get(i);
	    InputWeight.set(i, newWeight);
	}
    }

}

