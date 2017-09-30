/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Classifier.Tree;

import java.util.ArrayList;
import java.util.Enumeration;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author USER
 */
public class ID3Classifier extends AbstractClassifier {
    
    /** The training data used by the classifier */
    protected Instances trainingData;
    
    /** The root node of the classifier tree */
    protected ID3DecisionTree root;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
	// Remove instance with missing class value
	data.deleteWithMissingClass();
	
	if(data.classAttribute().isNumeric()) {
	    throw new Exception("Numeric Class attributes not supported");
	}
	Enumeration<Attribute> attributes = data.enumerateAttributes();
	while(attributes.hasMoreElements()) {
	    if(attributes.nextElement().isNumeric()) {
		throw new Exception("Numeric attributes not supported");
	    }
	}
	
        trainingData = data;
	root = new ID3DecisionTree(null, trainingData);
	setupTree(root);
	System.out.println(root);
    }
    
    /**
     * Recursive method that will select the best attribute to split the data at 
     * the given node, and call itself for each of the generated child node
     * @param node The node to be build
     * @throws Exception
     */
    protected void setupTree(ID3DecisionTree node) throws Exception {
	Instances data = node.getNodeData();
	if(!data.isEmpty()) {
	    int bestAttributeID = -1;
	    double bestInformationGain = 0;
	    
	    // Find the best attribute to be used at this node
	    for(int i=0; i<data.numAttributes(); i++) {
		if(i != data.classIndex()) {
		    
		    double currentInformationGain = countNominalInformationGain(data, i);
		    if(currentInformationGain > bestInformationGain) {
			bestInformationGain = currentInformationGain;
			bestAttributeID = i;
		    }
		    
		}
	    }
	    
	    // If there is an attribute that can be used to further classify the
	    // data, set it as this node splitter and setup each of its child
	    if(bestAttributeID != -1) {
		node.SetNominalSplitter(bestAttributeID);
		ArrayList<ID3DecisionTree> subTrees = node.getSubTrees();
		
		for(int i=0; i<subTrees.size(); i++) {
		    setupTree(subTrees.get(i));
		}
	    }
	}
    }
    
    /**
     * Count the nominal information gain for the given attributes
     * @throws Exception
     */
    private double countNominalInformationGain(Instances nodeData, int attID) throws Exception {
	Instances data = new Instances(nodeData);
	Attribute att = data.attribute(attID);
	Attribute classAtt = data.classAttribute();
	
	// The number of instances with [index] class value
	int[] classCount = new int[classAtt.numValues()];
	// The number of instances with [index-i] class value and [index-j] att value
	int[][] attClassCount = new int[classAtt.numValues()][att.numValues()];
	// The number of instances with [index] att value
	int[] attCount = new int[att.numValues()];
	
	// Count the data distributions
	Enumeration<Instance> instances = data.enumerateInstances();
	while(instances.hasMoreElements()) {
	    Instance instance = instances.nextElement();
	    classCount[(int)instance.classValue()] += 1;
	    if(instance.isMissing(att)) {
		throw new Exception("Missing value is not supported");
	    }
	    attClassCount[(int)instance.classValue()][(int)instance.value(attID)] += 1;
	    attCount[(int)instance.value(attID)] += 1;
	}
	
	// Calculate the class Attribute entropy
	double result = 0;
	for(int i=0; i<classCount.length; i++) {
	    double prob = (double)classCount[i] / (double)data.size();
	    if(prob > 0) {
		result -= prob * (Math.log10(prob) / Math.log10(2));
	    }
	}
	
	// Substract the class entropy with each partial entropy of att 
	for(int i=0; i<attCount.length; i++) {
	    for (int[] attClassCount1 : attClassCount) {
		double temp = 0;
		double prob = (double) attClassCount1[i] / (double)attCount[i];
		if(prob > 0) {
		    temp -= prob * (Math.log10(prob) / Math.log10(2));
		}
		result -= ((double)attCount[i] / (double)data.size()) * temp;
	    }
	}
	
	return result;
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
	
	if(!instance.equalHeaders(trainingData.firstInstance())) {
	    throw new Exception("Instance header is not equal to training data");
	}
		
	ID3DecisionTree currTree = root;
	while(true) {
	    
	    if(currTree.isLeaf) {
		double[] classDistribution = currTree.getClassDistribution();
		return classDistribution;
	    
	    } else {
		double[] subTreeDist = currTree.getSubTreeDistribution(instance);
		
		double max = 0;
		int maxID = -1;
		for(int i=0; i<subTreeDist.length; i++) {
		    if(subTreeDist[i] > max) {
			maxID = i;
			max = subTreeDist[i];
		    }
		}
		
		if(maxID != -1) {
		    currTree = currTree.getSubTrees().get(maxID);
		} else {
		    double[] classDistribution = currTree.getClassDistribution();
		    return classDistribution;
		}
	    }
	}
    }
    
}
