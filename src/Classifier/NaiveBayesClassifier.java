/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Classifier;

import java.util.Enumeration;
import java.util.Map;
import java.util.TreeMap;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

/**
 *
 * @author USER
 */
public class NaiveBayesClassifier extends AbstractClassifier {
    
    /** The discretize filter used to discretize the test data */
    private Discretize discretizeFilter;
    
    /** The training data used by the classifier */
    private Instances trainingData;

    /**
     * The model of the classifier, is a Three-Dimentional Map of 
     * ClassAttributeDomainValue -> NonClassAttribute -> NonClassAttributeDomainValue -> Probability
     */
    private final TreeMap<String,TreeMap<String,TreeMap<String,Float>>> ProbabilityMatrix = new TreeMap<>();
    
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
	
	// Discretize all the data attribute 
	discretizeFilter = new Discretize();
	discretizeFilter.setAttributeIndices("first-last");
	discretizeFilter.setBinRangePrecision(6);
	discretizeFilter.setInputFormat(data);
	trainingData = Filter.useFilter(data, discretizeFilter);
	
	
	// For each of the class attributes domain value in the instances,
	Attribute classAttribute = trainingData.classAttribute();
	Enumeration<Object> enumerateDomainValues = classAttribute.enumerateValues();
	while(enumerateDomainValues.hasMoreElements()) {
	    
	    // Prepare a sub-tree to contain each of the non-class atributes tree
	    String classValueString = enumerateDomainValues.nextElement().toString();
	    ProbabilityMatrix.put(classValueString, new TreeMap<>());
	    
	    // For each of the attributes in the instances,
	    Enumeration<Attribute> enumerateAttributes = trainingData.enumerateAttributes();
	    while(enumerateAttributes.hasMoreElements()) {

		// Prepare a sub-tree to contain each of this attributes domain value tree
		Attribute currAttributes = enumerateAttributes.nextElement();
		ProbabilityMatrix.get(classValueString).put(currAttributes.name(), new TreeMap<>());

		// For each of this attributes possible domain value
		Enumeration<Object> enumerateValues = currAttributes.enumerateValues();
		while(enumerateValues.hasMoreElements()) {

		    // Put the probability value for this classValue, attribute, and attValue
		    String currAttDomain = enumerateValues.nextElement().toString();
		    float prob = countProb(classValueString, currAttributes.name(), currAttDomain);
		    ProbabilityMatrix.get(classValueString).get(currAttributes.name()).put(currAttDomain, prob);

		}

	    }
	    
	}
	
	System.out.print(ProbabilityMatrix.toString());
	
    }
    
    private float countProb(String classValue, String attName, String attValue) {
	return Float.NaN;
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
	throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}