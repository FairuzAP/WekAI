/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Classifier;

import java.util.Random;
import java.util.Scanner;
import java.util.Vector;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;
/**
 *
 * @author USER
 */
public class FFNNClassifier extends AbstractClassifier {    

    private MultiLayerPerceptron MLP;
    private double learningRate = 0.1;
    Vector<Vector<Double>> dataInput = new Vector<>();

    /** A single perceptron container/helper class */
    private class Perceptron {

	/**
	 * The value and weight of all of this perceptron input 
	 * w0 is in index 0 and always had the value of 1
	 */
	private final Vector<Double> InputValue = new Vector<>();
	private final Vector<Double> InputWeight = new Vector<>();

	Random rand = new Random();


	/** Intiate the perceptron, add x0 with value 1 and a random weight */
	public Perceptron() {
	    rand.setSeed(System.currentTimeMillis());
	    InputValue.add(1.0);
	    InputWeight.add(rand.nextDouble());
	}

	
	/** Add another input with a random weight, return the new input weight */
	public int addInput(double value) {
	    InputValue.add(value);
	    InputWeight.add(rand.nextDouble());
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

	/** Improve each of the InputWeight entry according to the error value */
	public void learn(double rate, double error) {
	    for(int i=0; i<InputWeight.size(); i++) {
		double newWeight = InputWeight.get(i) + rate*error*InputValue.get(i);
		InputWeight.set(i, newWeight);
	    }
	}

    }

    /** A multi-layer perceptron container/helper class */
    private class MultiLayerPerceptron {
	
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

	/**
	 * Improve each of the InputWeight entry of each Perceptron.
	 * Err is the error for each of the last-layer perceptron.
	 * For the hidden layer, calculate the error with a sigmoid function
	 */
	public void backPropragate(double rate, Vector<Double> err) {
	    
	    // Prepare the vector containing each perceptron's error
	    Vector<Vector<Double>> MLPErr = new Vector<>();
	    MLPErr.add(err);
	    
	    // For each layer other than the output layer, starting from the back
	    for(int i=MLP.size()-2; i>=0; i--) {

                // Insert the next layer from the front
                MLPErr.insertElementAt(new Vector<>(),0);

                // For every perceptron in this layer
                for(int j=0; j<MLP.get(i).size(); j++) {

                    // Calculate the error
                    double out = MLP.get(i).get(j).getOutput();
                    double error = 0.0;

                    // For every perceptron in the next layer
                    for(int k=0; k<MLP.get(i+1).size(); k++) {

                        // Add the weight of the input from the calculated perceptron
                        // times the error of this perceptron
                        error+= MLP.get(i+1).get(k).getWeight(j+1) * MLPErr.get(1).get(k);

                    }

                    error *= out*(1-out);
                    MLPErr.get(0).add(error);
                }
	    }
	    // Update the weight of all perceptron from the front
	    for(int i=0; i<MLP.size(); i++) {
		for(int j=0; j<MLP.get(i).size(); j++) {
		    MLP.get(i).get(j).learn(rate, MLPErr.get(i).get(j));
		}
	    }
	}
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
        int maxEpoch = 10;
        int epoch = 0;
        double target = 0.5;
        double epochError = target + 1;

        int hiddenCount = 1;
        Vector<Integer> perceptronCount = new Vector<>();

        int inputCount = data.numAttributes();
//        int outputCount = data.numClasses();
        Scanner in = new Scanner(System.in);
        System.out.println("masukkan jumlah node pada hidden layer : ");
        int x = in.nextInt();
        perceptronCount.add(inputCount);
        perceptronCount.add(x);
        perceptronCount.add(data.numClasses());
        MLP = new MultiLayerPerceptron(hiddenCount, perceptronCount);

        for (int i = 0; i < inputCount; i++) {
            dataInput.add(new Vector<>());
        }

        for (int i = 0; i < data.numInstances(); i++) {
            Instance currData = data.get(i);
            for (int j = 0; j < inputCount; j++) {
                dataInput.get(j).add(currData.value(j));
            }
        }

        do {
            epoch++;
            for (int i = 0; i < data.numInstances(); i++) {
                MLP.setInputs(dataInput.get(i));
                MLP.frontPropragate();
                MLP.backPropragate(learningRate, errorCount());
            }
            epochError = epochErrorCount(data);
        } while ((epochError >= target) && (epoch < maxEpoch));
    }
    
    /**
     * Classifies the given test instance. The instance has to belong to a dataset
     * when it's being classified. Note that a classifier MUST implement either
     * this or distributionForInstance().
     *
     * @param instance the instance to be classified
     * @return the predicted most likely class for the instance or
     *         Utils.missingValue() if no prediction is made
     * @exception Exception if an error occurred during the prediction
     */
    @Override
    public double classifyInstance(Instance instance) throws Exception {
	return 0;
    }

    private Vector<Double> errorCount() {
        Vector<Double> output = MLP.getOutputs();
        double max = output.get(0);
        for (int i = 1; i < output.capacity(); i++) {
            if (max < output.get(i)) {
                max = output.get(i);
            }
        }
        Vector<Double> errorClass = new Vector<>();
        for (int i = 0; i < output.capacity(); i++) {
            double err;
            if (max == output.get(i)) {
                double outVal = output.get(i);
                err = outVal * (1 - outVal) * (1 - outVal);
            } else {
                double outVal = output.get(i);
                err = outVal * (1 - outVal) * (0 - outVal);
            }
            errorClass.add(err);
        }
        return errorClass;
    }

    private double epochErrorCount(Instances data) {
        double error = 0;
        Vector<Double> errorClass = new Vector<>();
        for (int i = 0; i < data.numInstances(); i++) {
            MLP.setInputs(dataInput.get(i));
            MLP.frontPropragate();
            errorClass = errorCount();
            for (int j = 0; j < errorClass.capacity(); j++) {
                error = error + errorClass.get(j);
            }
        }
        return error;
    }

}
