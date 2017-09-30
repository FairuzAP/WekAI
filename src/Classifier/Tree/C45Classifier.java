/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Classifier.Tree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Map;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import javafx.util.Pair;


/**
 *
 * @author USER
 */
public class C45Classifier extends ID3Classifier {
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
	// Remove instance with missing class value
        trainingData = data;
	trainingData.deleteWithMissingClass();

	root = new C45DecisionTree(null, trainingData);
	setupTree(root);
	System.out.println(root);
    }
    
    /**
     * Recursive method that will select the best attribute to split the data at 
     * the given node, and call itself for each of the generated child node
     * @param node The node to be build
     * @throws Exception
     */
    @Override
    protected void setupTree(ID3DecisionTree node) throws Exception {
	Instances data = node.getNodeData();
	if(!data.isEmpty()) {
	    int bestAttributeID = -1;
	    double bestNumericBoundaries[] = null;
	    double bestInformationGain = 0;
	    
	    // Find the best attribute to be used at this node
	    for(int i=0; i<data.numAttributes(); i++) {
		if(i != data.classIndex()) {
		    double currentInformationGain = 0.0;
		    double currentNumericBoundaries[] = null;
		    
		    if(data.attribute(i).isNominal()) {
			currentInformationGain = countNominalGainRatio(data, i);
		    } else {
			Pair<Double, Double> res = getNumericInformationGain(data, i);
			currentInformationGain = res.getKey();
			currentNumericBoundaries = new double[]{res.getValue()};
		    }
		    
		    if(currentInformationGain > bestInformationGain) {
			bestInformationGain = currentInformationGain;
			bestAttributeID = i;
			if(data.attribute(bestAttributeID).isNumeric()) {
			    bestNumericBoundaries = currentNumericBoundaries;
			}
		    }
		}
	    }
	    
	    // If there is an attribute that can be used to further classify the
	    // data, set it as this node splitter and setup each of its child
	    if(bestAttributeID != -1) {
		if(data.attribute(bestAttributeID).isNominal()) {
		    node.SetNominalSplitter(bestAttributeID);
		} else {
		    ((C45DecisionTree)node).SetNumericSplitter(bestAttributeID, bestNumericBoundaries);
		}
		
		ArrayList<ID3DecisionTree> subTrees = node.getSubTrees();
		for(int i=0; i<subTrees.size(); i++) {
		    setupTree(subTrees.get(i));
		}
	    }
	}
    }
    
    /**
     * Recursively trim the tree starting at the bottom using the supplied test data
     * @param testData The validation data used for trimming
     * @throws java.lang.Exception
     */
    protected void trimTree(Instances testData) throws Exception {
	trimTree((C45DecisionTree) root, testData);
    }
    private void trimTree(C45DecisionTree node, Instances testData) throws Exception {
	if(!node.isLeaf()) {
	    for(int i=0; i<node.getSubTrees().size(); i++) {
		trimTree((C45DecisionTree) node.getSubTrees().get(i), testData);
	    }
	    
	    Evaluation old_eval = new Evaluation(trainingData);
	    old_eval.evaluateModel(this, testData);
	    node.isLeaf = true;
	    Evaluation new_eval = new Evaluation(trainingData);
	    new_eval.evaluateModel(this, testData);
	    
	    if(old_eval.errorRate() < new_eval.errorRate()) {
		node.isLeaf = false;
	    }
	}
    }
    
    /**
     * Count the nominal gain ratio for the given nominal attributes
     * @throws Exception
     */
    private double countNominalGainRatio(Instances nodeData, int attID) throws Exception {
	Instances data = new Instances(nodeData);
	Attribute att = data.attribute(attID);
	Attribute classAtt = data.classAttribute();
	data.deleteWithMissing(attID);
	
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
	
	// Calculate the chosen attribute entropy
	double attEntropy = 0;
	for(int i=0; i<attCount.length; i++) {
	    double prob = (double)attCount[i] / (double)data.size();
	    if(prob > 0) {
		attEntropy -= prob * (Math.log10(prob) / Math.log10(2));
	    }
	}
	
	return result/attEntropy;
    }
    
    /**
     * Count the information gain for the given numeric attributes, and
     * return a tuple of the IG and splitter point
     */
    private Pair<Double,Double> getNumericInformationGain(Instances nodeData, int attID) {
	Instances data = new Instances(nodeData);
	Attribute att = data.attribute(attID);
	Attribute classAtt = data.classAttribute();
	data.deleteWithMissing(attID);
	data.sort(att);
	
	// The number of instances with [index] class value
	int[] classCount = new int[classAtt.numValues()];
	
	class StatHolder {
	    // The number of instances with attValue < splitPoint and [index] class value
	    public int[][] attClassCount;
	    public int[] attCount;
	    
	    @Override
	    public String toString() {
		try {
		    return Arrays.deepToString(attClassCount);
		} catch (Exception ex) {
		    Logger.getLogger(ID3DecisionTree.class.getName()).log(Level.SEVERE, null, ex);
		    return "";
		}
	    }
	}
	
	// Mapping of potential splitPoints and it stats
	TreeMap<Double, StatHolder> dataStats = new TreeMap<>();
	double lastClass = data.firstInstance().classValue();
	StatHolder next = new StatHolder();
	next.attClassCount = new int[classAtt.numValues()][2];
	next.attCount = new int[2];
	dataStats.put(data.firstInstance().value(att), next);
	
	// Count the data distributions
	Enumeration<Instance> instances = data.enumerateInstances();
	while(instances.hasMoreElements()) {
	    Instance instance = instances.nextElement();
	    double currClass = instance.classValue();
	    double currAtt = instance.value(att);
	    classCount[(int)currClass] += 1;
	    
	    if(currClass != lastClass) {
		next = new StatHolder();
		next.attClassCount = new int[classAtt.numValues()][2];
		next.attCount = new int[2];
		StatHolder prev = dataStats.get(dataStats.floorKey(currAtt));
		for(int i=0; i<classAtt.numValues(); i++) {
		    next.attClassCount[i][0] = prev.attClassCount[i][0]+prev.attClassCount[i][1];
		    next.attCount[0] += prev.attClassCount[i][0]+prev.attClassCount[i][1];
		}
		dataStats.put(currAtt, next);
	    }
	    
	    dataStats.entrySet().stream().forEach((entry) -> {
		entry.getValue().attClassCount[(int)currClass][1] += 1;
		entry.getValue().attCount[1] += 1;
	    });
	    lastClass = currClass;
	}
	
	// Calculate the class Attribute entropy
	double result = 0;
	for(int i=0; i<classCount.length; i++) {
	    double prob = (double)classCount[i] / (double)data.size();
	    if(prob > 0) {
		result -= prob * (Math.log10(prob) / Math.log10(2));
	    }
	}
	
	double bestIG = 0;
	double bestSplitter = -1;
	
	for(Map.Entry<Double, StatHolder> entry : dataStats.entrySet()) {
	    
	    StatHolder value = entry.getValue();
	    double tempResult = result;
	    
	    // Substract the class entropy with each partial entropy of att 
	    for(int i=0; i<value.attCount.length; i++) {
		for (int[] attClassCount1 : value.attClassCount) {
		    double temp = 0;
		    double prob = (double) attClassCount1[i] / (double)value.attCount[i];
		    if(prob > 0) {
			temp -= prob * (Math.log10(prob) / Math.log10(2));
		    }
		    tempResult -= ((double)value.attCount[i] / (double)data.size()) * temp;
		}
	    }
	    
	    if(tempResult > bestIG) {
		bestIG = tempResult;
		bestSplitter = entry.getKey();
	    }
	    
	}
	
	Pair<Double,Double> res = new Pair<>(bestIG, bestSplitter);
	return res;
    }
    
}
