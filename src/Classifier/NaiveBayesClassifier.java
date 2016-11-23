/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Classifier;

import java.util.Enumeration;
import java.util.TreeMap;

import weka.classifiers.AbstractClassifier;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveWithValues;


/**
 *
 * @author USER
 */
public class NaiveBayesClassifier extends AbstractClassifier {
    
    /** The discretize removeFilter used to discretize the test data */
    private Discretize discretizeFilter;
    
    /** The normalize filter for the test data */
    private Normalize normalFilter;

    /** The training data used by the classifier */
    private Instances trainingData;

    /**
     * The model of the classifier, is a Three-Dimentional Map of 
     * ClassAttributeDomainValue -> NonClassAttribute -> NonClassAttributeDomainValue -> Probability
     */
    private final TreeMap<String,TreeMap<String,TreeMap<String,Float>>> ProbabilityMatrix = new TreeMap<>();
    /**
     * A map between a class attribute value and the probabilty of that value
     */
    private final TreeMap<String,Float> ProbMatrixClass = new TreeMap<>();
    
    
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
	
	// Remove instance with missing class value
	data.deleteWithMissingClass();
	
	/** Remove unwanted class attr */
	Remove remove = new Remove();
	remove.setAttributeIndices("28");
	remove.setInputFormat(data);
	Instances tempData = Filter.useFilter(data, remove);

	// Discretize all the data attribute 
	discretizeFilter = new Discretize();
	discretizeFilter.setAttributeIndices("first-last");
	discretizeFilter.setBinRangePrecision(6);
	discretizeFilter.setInputFormat(tempData);

        trainingData = Filter.useFilter(tempData, discretizeFilter);

	RemoveWithValues removeFilter = new RemoveWithValues();
	removeFilter.setInvertSelection(true);
	
	// For each of the class attributes domain value in the instances,
	Attribute classAttribute = trainingData.classAttribute();
	Enumeration<Object> enumerateDomainValues = classAttribute.enumerateValues();
	while(enumerateDomainValues.hasMoreElements()) {
	    
	    // Prepare a sub-tree to contain each of the class attribute value
	    String classValueString = enumerateDomainValues.nextElement().toString();
	    ProbabilityMatrix.put(classValueString, new TreeMap<>());
	    
	    // Get a sub-data where the class-value is classValueString
	    removeFilter.setAttributeIndex("" + (trainingData.classIndex()+1));
	    removeFilter.setNominalIndices("" + (classAttribute.indexOfValue(classValueString)+1));
	    removeFilter.setInputFormat(trainingData);
	    Instances subDataClassValue = Filter.useFilter(trainingData, removeFilter);
	    
	    // Insert the probability of this class values
	    float prob2 = (float)subDataClassValue.size() / trainingData.size();
	    ProbMatrixClass.put(classValueString, prob2);
	    
	    // For each of the attributes in the instances,
	    Enumeration<Attribute> enumerateAttributes = trainingData.enumerateAttributes();
	    while(enumerateAttributes.hasMoreElements()) {

		// Prepare a sub-tree to contain each of this attributes domain value tree
		Attribute currAttributes = enumerateAttributes.nextElement();
		ProbabilityMatrix.get(classValueString).put(currAttributes.name(), new TreeMap<>());

		// For each of this attributes possible domain value
		Enumeration<Object> enumerateValues = currAttributes.enumerateValues();
		while(enumerateValues.hasMoreElements()) {
		    
		    // Insert the probability of this attributes values at the current class att values
		    String currAttDomain = enumerateValues.nextElement().toString();
		    
		    // Get a sub-data where the currAttributes value is currAttDomain
		    removeFilter.setAttributeIndex("" + (currAttributes.index()+1) );
		    removeFilter.setNominalIndices("" + (currAttributes.indexOfValue(currAttDomain)+1) );
		    removeFilter.setInputFormat(subDataClassValue);
		    Instances subDataAttValues = Filter.useFilter(subDataClassValue, removeFilter);
		    
		    float prob = (float)subDataAttValues.size() / subDataClassValue.size();
		    ProbabilityMatrix.get(classValueString).get(currAttributes.name()).put(currAttDomain, prob);

		}

	    }
	    
	}
	
    }
    
    
    /**
     * Return the partial probability of the attributeName value to be 
     * attribute value if the class attribute value is classDomainValue
     */
    private float getPartialProbability(String classDomainValue, String attributeName, String attributeValue) {
	return ProbabilityMatrix.get(classDomainValue).get(attributeName).get(attributeValue);
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
	
	// Preprocess the instance (discretize using the same filter)
	Instances Result = new Instances(trainingData,0);
	Result.setClassIndex(trainingData.classIndex());
	Result.add(instance);
	Result = Filter.useFilter(Result, discretizeFilter);
	
	// Initialize the result array by assigning the probability of each class value
	double[] prob = new double[Result.classAttribute().numValues()];
	for(int i=0; i<Result.classAttribute().numValues(); i++) {
	    prob[i] = ProbMatrixClass.get(Result.classAttribute().value(i));
	}
	
	// For all the Non-Class attribute in the instance,
	Enumeration<Attribute> enumerateAttributes = Result.enumerateAttributes();
	while(enumerateAttributes.hasMoreElements()) {
	    
	    // For all the possible class domain value,
	    Attribute currAtt = enumerateAttributes.nextElement();
	    for(int i=0; i<Result.classAttribute().numValues(); i++) {
		
		String classValue = Result.classAttribute().value(i);
		String attName = currAtt.name();
		String attValue = currAtt.value((int) Result.firstInstance().value(currAtt));
		prob[i] *= getPartialProbability(classValue, attName, attValue);
	    
	    }
	    
	}
	
	return prob;
    }
    
}