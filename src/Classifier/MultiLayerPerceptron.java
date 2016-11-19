/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Classifier;

import java.io.Serializable;
import java.util.Vector;

/** A multi-layer perceptron container/helper class */
class MultiLayerPerceptron implements Serializable {

    /**
     * An array of Perceptron layer, The output is the layer at the last index
     * Anything else is a hidden layer
     */
    private final Vector<Vector<Perceptron>> MLP = new Vector<>();


    /**
     * An MLP constructor
     * @param hidLayerCount The amount of hidden layer in this MLP
     * @param perceptronCount A vector with size hidLayerCount+2, the 0th 
     *  index contains the input-count, the last index contains the output
     *  perceptron count, the rest is hidden layer perceptron count
     */
    public MultiLayerPerceptron(int hidLayerCount, Vector<Integer> perceptronCount) {

	// For each requested layer; including the output layer..
	// Skip the first index because it contains the input count
	for(int i=1; i<=hidLayerCount+1; i++) {
	    MLP.add(new Vector<>());

	    // Create the requested perceptron in this layer
	    for(int j=0; j<perceptronCount.get(i); j++) {
		MLP.get(i-1).add(new Perceptron());

		// Prepare the input from the previous layer. In case of
		// first layer, will fetch perceptronCount[0], which is 
		// the input count
		for(int k=0; k<perceptronCount.get(i-1); k++) {
		    MLP.get(i-1).get(j).addInput(0);
		}
	    }
	}
    }

    /** 
     * Set the input value for each of the first layer perceptron
     * The size of vector must equal to the inputCount supplied at constructor
     */
    public void setInputs(Vector<Double> Input) {
	int i = 1;
	for(double d : Input) {
	    for(int j=0; j<MLP.get(0).size(); j++) {
		MLP.get(0).get(j).setValue(i, d);
	    }
	    i++;
	}
    }

    /** 
     * Front-propragate the input value to the last layer 
     * @return the output of the last perceptron layer after propragation
     */
    public Vector<Double> frontPropragate() {
	//Looping layer
	for (int i = 0; i < MLP.size()-1; i++) {

	    //Looping neuron
	    for (int j = 0; j < MLP.get(i).size(); j++){
		double o = MLP.get(i).get(j).getOutput();

		for (int k = 0; k < MLP.get(i+1).size(); k++){
		    MLP.get(i+1).get(k).setValue(j+1, o);
		}

	    }

	}
	return getOutputs();
    }

    /**
     * Return the output OF ALL perceptron in the last layer with a sigmoid 
     * activation function
     */
    public Vector<Double> getOutputs() {
	Vector<Double> res = new Vector<>();
	for(int i=0; i<MLP.get(MLP.size()-1).size(); i++) {
	    res.add(MLP.get(MLP.size()-1).get(i).getOutput());
	}
	return res;
    }

    public Vector<Double> getRawOutputs() {
	Vector<Double> res = new Vector<>();
	for(int i=0; i<MLP.get(MLP.size()-1).size(); i++) {
	    res.add(MLP.get(MLP.size()-1).get(i).getRawOutput());
	}
	return res;
    }
    
    /**
     * Improve each of the InputWeight entry of each Perceptron.
     * Err is the totError for each of the last-layer perceptron.
     * For the hidden layer, calculate the totError with a sigmoid function
     */
    public void backPropragate(double rate, Vector<Double> err) {

	// Prepare the vector containing each perceptron's totError
	Vector<Vector<Double>> MLPErr = new Vector<>();
	MLPErr.add(err);

	// For each layer other than the output layer, starting from the back
	for(int i=MLP.size()-2; i>=0; i--) {

	    // Insert the next layer from the front
	    MLPErr.insertElementAt(new Vector<>(),0);

	    // For every perceptron in this layer
	    for(int j=0; j<MLP.get(i).size(); j++) {

		// Calculate the totError
		double out = MLP.get(i).get(j).getOutput();
		double error = 0.0;

		// For every perceptron in the next layer
		for(int k=0; k<MLP.get(i+1).size(); k++) {

		    // Add the weight of the input from the calculated perceptron
		    // times the totError of this perceptron
		    error+= MLP.get(i+1).get(k).getWeight(j+1) * MLPErr.get(1).get(k);

		}

		error *= out*(1-out);
		MLPErr.get(0).add(error);
	    }
	}
	int k = 0;
	// Update the weight of all perceptron from the front
	for(int i=0; i<MLP.size(); i++) {
	    for(int j=0; j<MLP.get(i).size(); j++) {
		MLP.get(i).get(j).learn(rate, MLPErr.get(i).get(j));
	    }
	}
    }
}

