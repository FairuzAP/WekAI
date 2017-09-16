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
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

/**
 *
 * @author USER
 */
public class ID3Classifier extends AbstractClassifier {
    
    /** The discretize removeFilter used to discretize the test data */
    private Discretize discretizeFilter;
    
    /** The training data used by the classifier */
    private Instances trainingData;
    
    private ID3DecisionTree root;
    
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
	root = new ID3DecisionTree(null, trainingData);
	setupTree(root);
	System.out.println(root);
    }
    
    protected void setupTree(ID3DecisionTree node) throws Exception {
	Instances data = node.getNodeData();
	if(!data.isEmpty()) {
	    int bestAttributeID = -1;
	    double bestInformationGain = 0;
	    
	    for(int i=0; i<data.numAttributes(); i++) {
		if(i != data.classIndex()) {
		    
		    double currentInformationGain = countNominalInformationGain(data, i);
		    if(currentInformationGain > bestInformationGain) {
			bestInformationGain = currentInformationGain;
			bestAttributeID = i;
		    }
		    
		}
	    }
	    
	    if(bestAttributeID != -1) {
		node.SetNominalSplitter(bestAttributeID);
		ArrayList<ID3DecisionTree> subTrees = node.getSubTrees();
		
		for(int i=0; i<subTrees.size(); i++) {
		    setupTree(subTrees.get(i));
		}
	    }
	}
    }
    
    protected final double countNominalInformationGain(Instances nodeData, int attID) throws Exception {
	Instances data = new Instances(nodeData);
	Attribute att = data.attribute(attID);
	Attribute classAtt = data.classAttribute();
	data.deleteWithMissing(attID);
	
	int[] classCount = new int[classAtt.numValues()];
	int[][] attClassCount = new int[classAtt.numValues()][att.numValues()];
	int[] attCount = new int[att.numValues()];
	
	Enumeration<Instance> instances = data.enumerateInstances();
	while(instances.hasMoreElements()) {
	    Instance instance = instances.nextElement();
	    classCount[(int)instance.classValue()] += 1;
	    attClassCount[(int)instance.classValue()][(int)instance.value(attID)] += 1;
	    attCount[(int)instance.value(attID)] += 1;
	}
	
	double result = 0;
	
	for(int i=0; i<classCount.length; i++) {
	    double prob = (double)classCount[i] / (double)data.size();
	    if(prob > 0) {
		result -= prob * (Math.log10(prob) / Math.log10(2));
	    }
	}
	
	for(int i=0; i<attCount.length; i++) {
	    for(int j=0; j<attClassCount.length; j++) {
		double temp = 0;
		double prob = (double)attClassCount[j][i] / (double)attCount[i];
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
