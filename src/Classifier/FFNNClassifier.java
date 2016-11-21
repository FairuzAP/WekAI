/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Classifier;

import java.io.Serializable;
import static java.lang.Double.NaN;
import java.util.Enumeration;
import java.util.Random;
import java.util.Scanner;
import java.util.Vector;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.instance.RemoveWithValues;

/**
 *
 * @author USER
 */
public class FFNNClassifier extends AbstractClassifier {
    
    /** The training data used by the classifier */
    private Instances trainingData;
    
    /** Normalize Filter for FFNN*/
    Normalize normalFilter;
    
    /** The MLP-model used by this classifier */
    private MultiLayerPerceptron MLP;
    
    /** Learning paramater and stop condition */
    private double learningRate = 0.3;
    private int maxEpoch = 2000;
    private double target = 0;
    
    /** The vector containing how many perceptron in each requested hidden layer */
    private Vector<Integer> perceptronCount = new Vector<>();
    

    /**
     * Generates a classifier. Must initialize all fields of the classifier that 
     * are not being set via options (ie. multiple calls of buildClassifier must 
     * always lead to the same result). Must not change the dataset in any way.
     *
     * @param data set of instances serving as training trainingData
     * @exception Exception if the classifier has not been generated successfully
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
	
	// Data initialization and reading
        normalFilter = new Normalize();
	normalFilter.setScale(6);
	normalFilter.setTranslation(-3);
	normalFilter.setInputFormat(data);
	trainingData = Filter.useFilter(data, normalFilter);
	trainingData.deleteWithMissingClass();
	
	normalFilter = new Normalize();
	normalFilter.setScale(4);
	normalFilter.setTranslation(-2);
	normalFilter.setInputFormat(data);
	trainingData = Filter.useFilter(data, normalFilter);
	
	int hiddenCount = perceptronCount.size(); 
        int inputCount = trainingData.numAttributes() - 1;
	perceptronCount.insertElementAt(inputCount, 0);
	int outputCount = trainingData.classAttribute().numValues();
	perceptronCount.add(outputCount);
	
	// Initialize the MLP-model
        MLP = new MultiLayerPerceptron(hiddenCount, perceptronCount);
	
	// Loop to improve MLP model
	int epoch = 0;
	double epochError;
        do {
            epoch++;
	    
	    // For ever instance in the training data
            for (int i = 0; i < trainingData.numInstances(); i++) {
		
		// Store every non-class attribute value as double of this instances to a vector
		Vector<Double> in = new Vector<>(); 
		Enumeration<Attribute> enumerateAttributes = trainingData.enumerateAttributes();
		while(enumerateAttributes.hasMoreElements()) {
		    in.add(trainingData.get(i).value(enumerateAttributes.nextElement()));
		}
		
		// Propagate the input and learn based on the error value
                MLP.setInputs(in);
                MLP.frontPropragate();
                MLP.backPropragate(learningRate, errorCount(i));
		
            }
            epochError = epochErrorCount();
        } while ((epochError > target) && (epoch < maxEpoch));
	int i = 1;
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
	
	// Preprocess the instance 
	Instances newInstances = new Instances(trainingData,0);
	newInstances.setClassIndex(trainingData.classIndex());
	newInstances.add(instance);
	newInstances = Filter.useFilter(newInstances, normalFilter);
	
	// Store every non-class attribute value as double of this instances to a vector
	Vector<Double> in = new Vector<>(); 
	Enumeration<Attribute> enumerateAttributes = newInstances.enumerateAttributes();
	while(enumerateAttributes.hasMoreElements()) {
	    in.add(newInstances.get(0).value(enumerateAttributes.nextElement()));
	}
	
	// Propagate the input through the learned MLP-model
	MLP.setInputs(in);
	MLP.frontPropragate();
	Vector<Double> frontPropragate = MLP.getOutputs();
		
	// Search for the index of output perceptron with the highest value
	double iMax = 0;
	double vMax = 0;
	for(int i=0; i<frontPropragate.size(); i++) {
	    if(frontPropragate.get(i)>vMax) {
		iMax = i;
		vMax = frontPropragate.get(i);
	    }
	}
	
	double[] prob = new double[instance.classAttribute().numValues()];
	for(int i=0; i<instance.classAttribute().numValues(); i++) {
	    if (i == iMax) {
		prob[i] = 1;
	    } else {
		prob[i] = 0;
	    }
	}
	return prob;
    }
    
    private Vector<Double> errorCount(int instanceIndex) {
	
	// Fetch the output and maxOutput of the MLP 
        Vector<Double> output = MLP.getOutputs();
	double targetIdx = trainingData.get(instanceIndex).value(trainingData.classIndex());
	
	// NEED TO HANDLE IF ONLY ONE OUTPUT PERCEPTRON,
	
        Vector<Double> errorClass = new Vector<>();
        for (int i=0; i < output.size(); i++) {
            double err;
	    double outVal = output.get(i);
	    // outVal*(1-outVal)*
            if (i == targetIdx) {
                err = outVal*(1-outVal)*(1 - outVal);
            } else {
                err = outVal*(1-outVal)*(0 - outVal);
            }
            errorClass.add(err);
        }
	
        return errorClass;
    }
    
    /**
     * Count and return the sum of Mean absolute error the current MLP produce
     * for the training data
     */
    private double epochErrorCount() {
	
	// Variables Initialization
        double totError = 0;
        Vector<Double> errorClass = new Vector<>();
        
	// For all instance in the traningData,
	for (int i=0; i < trainingData.numInstances(); i++) {
	    
	    // Store every non-class attribute value as double of this instances to a vector
	    Vector<Double> in = new Vector<>(); 
	    Enumeration<Attribute> enumerateAttributes = trainingData.enumerateAttributes();
	    while(enumerateAttributes.hasMoreElements()) {
		in.add(trainingData.get(i).value(enumerateAttributes.nextElement()));
	    }
	    
	    // Front propagate the value of this instance
            MLP.setInputs(in);
            MLP.frontPropragate();
	    
	    // Count the error and add the absolute of all error to the totError
            errorClass = errorCount(i);
            for (int j=0; j < errorClass.size(); j++) {
                totError = totError + Math.abs(errorClass.get(j));
            }
        }
	
        return totError;
    }
    
    
    public void setLearningRate(double rate) {
	learningRate = rate;
    }
    public void setMaxEpoch (int max) {
	maxEpoch = max;
    }
    public void setTarget (double t) {
	target = t;
    }
    public void setPerceptronCount (Vector<Integer> pCount) {
	perceptronCount = new Vector<>(pCount);
    }
    
}
