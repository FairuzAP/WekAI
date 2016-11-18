/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Classifier;

import java.util.Enumeration;
import java.util.Random;
import java.util.Scanner;
import java.util.Vector;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;
/**
 *
 * @author USER
 */
public class FFNNClassifier extends AbstractClassifier {    

    
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

	/** Improve each of the InputWeight entry according to the totError value */
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
	    // Update the weight of all perceptron from the front
	    for(int i=0; i<MLP.size(); i++) {
		for(int j=0; j<MLP.get(i).size(); j++) {
		    MLP.get(i).get(j).learn(rate, MLPErr.get(i).get(j));
		}
	    }
	}
    }
    
    /** The training data used by the classifier */
    private Instances trainingData;
    
    /** The MLP-model used by rhis classifier */
    private MultiLayerPerceptron MLP;
    
    /** Learning paramater and stop condition */
    private double learningRate = 0.1;
    int maxEpoch = 10;
    double target = 0.5;
    
    /** The vector containing how many perceptron in each requested hidden layer */
    Vector<Integer> perceptronCount = new Vector<>();
    
    // ???
    Vector<Vector<Double>> dataInput;
    
    
    /**
     * Generates a classifier. Must initialize all fields of the classifier that 
     * are not being set via options (ie. multiple calls of buildClassifier must 
     * always lead to the same result). Must not change the dataset in any way.
     *
     * @param trainingData set of instances serving as training trainingData
     * @exception Exception if the classifier has not been generated successfully
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
	
	// Data initialization and reading
	trainingData = new Instances(data);
	int hiddenCount = perceptronCount.size(); 
        int inputCount = trainingData.numAttributes() - 1;
	perceptronCount.insertElementAt(inputCount, 0);
	int outputCount = trainingData.classAttribute().numValues();
	perceptronCount.add(outputCount);
	
	// Initialize the MLP-model
        MLP = new MultiLayerPerceptron(hiddenCount, perceptronCount);
	
	// hmm,.. redundan sih dataInput itu, nggak bisa diakses berdasarkan konteks juga
        for (int i=0; i < inputCount; i++) {
            dataInput.add(new Vector<>());
        }	
        for (int i = 0; i < trainingData.numInstances(); i++) {
            Instance currData = trainingData.get(i);
            for (int j = 0; j < inputCount; j++) {
                dataInput.get(j).add(currData.value(j));
            }
        }
	
	int epoch = 0;
	double epochError;
        do {
            epoch++;
            for (int i = 0; i < trainingData.numInstances(); i++) {
                MLP.setInputs(dataInput.get(i));
                MLP.frontPropragate();
                MLP.backPropragate(learningRate, errorCount());
            }
            epochError = epochErrorCount();
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
     * @throws Exception if an totError occurred during the prediction
     */
    @Override
    public double classifyInstance(Instance instance) throws Exception {
	
	// Preprocess the instance 
	Instances newInstances = new Instances(trainingData,0);
	newInstances.setClassIndex(trainingData.classIndex());
	newInstances.add(instance);
	
	// Store every non-class attribute value as double of this instances to a vector
	Vector<Double> in = new Vector<>(); 
	Enumeration<Attribute> enumerateAttributes = newInstances.enumerateAttributes();
	while(enumerateAttributes.hasMoreElements()) {
	    in.add(newInstances.get(0).value(enumerateAttributes.nextElement()));
	}
	
	// Propagate the input through the learned MLP-model
	MLP.setInputs(in);
	Vector<Double> frontPropragate = MLP.frontPropragate();
	
	// Search for the index of output perceptron with the highest value
	double iMax = 0;
	double vMax = 0;
	for(int i=0; i<frontPropragate.size(); i++) {
	    if(frontPropragate.get(i)>vMax) {
		iMax = i;
		vMax = frontPropragate.get(i);
	    }
	}
	
	return iMax;
    }
    
    private Vector<Double> errorCount() {
	
	// Fetch the output and maxOutput of the MLP 
        Vector<Double> output = MLP.getOutputs();
        double max = output.get(0);
        for (int i=1; i < output.size(); i++) {
            if (max < output.get(i)) {
                max = output.get(i);
            }
        }
	
	// NEED TO HANDLE IF ONLY ONE OUTPUT PERCEPTRON,
	
	// ... J, ini kamu tahu target-nya dari mana?..
        Vector<Double> errorClass = new Vector<>();
        for (int i=0; i < output.size(); i++) {
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
    
    /**
     * Count and return the sum of half-squared error the current MLP produce
     * for the training data
     */
    private double epochErrorCount() {
	
	// Variables Initialization
        double totError = 0;
        Vector<Double> errorClass = new Vector<>();
        
	// For all instance in the traningData,
	for (int i=0; i < trainingData.numInstances(); i++) {
	    
	    // Front propagate the value of this instance
            MLP.setInputs(dataInput.get(i));
            MLP.frontPropragate();
	    
	    // Count the totError and add the ((error)^2)/2 of all error to the totError
            errorClass = errorCount();
            for (int j=0; j < errorClass.size(); j++) {
                totError = totError + Math.pow(errorClass.get(j),2)/2;
            }
	    
        }
	
        return totError;
    }

}
