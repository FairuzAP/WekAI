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
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

/**
 * Model Class for a Node in a ID3 decision Tree,
 * Assuming all instances is nominal, may behave abnormally otherwise
 * 
 * @author USER
 */
public class ID3DecisionTree {
    
    protected final ID3DecisionTree parent;
    protected final Instances nodeTrainingData;
    
    protected boolean isLeaf;
    protected ArrayList<ID3DecisionTree> subTrees;
    
    /** The attribute ID used to separate the data at this node */
    protected int splitterAttributeID;
    protected TreeMap<String, Integer> nominalSplitter;
    
    
    /** Make a new tree node with the supplied training data */
    public ID3DecisionTree(ID3DecisionTree parentNode, Instances subData) {
	isLeaf = true;
	splitterAttributeID = -1;
	nodeTrainingData = subData;
	nominalSplitter = new TreeMap<>();
	subTrees = new ArrayList<>();
	parent = parentNode;
    }
    
    /**
     * Will setup the splitter and initialize the subtree of this node by using
     * the attID as the splitter attribute
     * @param attID The attribute ID this node will use to split the data
     * @throws java.lang.Exception
     */
    public void SetNominalSplitter(int attID) throws Exception {
	nominalSplitter.clear();
	subTrees.clear();
	
	isLeaf = false;
	Attribute splitter = nodeTrainingData.attribute(attID);
	splitterAttributeID = attID;

	Enumeration<Object> values = splitter.enumerateValues();
	int i = 0;

	while(values.hasMoreElements()) {
	    String value = values.nextElement().toString();
	    nominalSplitter.put(value, i);
	    subTrees.add(null);
	    i++;
	}

	setupSubTrees();
    }
    
    /**
     * Will initialize the subtree according to the splitter
     * @throws Exception
     */
    protected void setupSubTrees() throws Exception {
	RemoveWithValues removeFilter = new RemoveWithValues();
	removeFilter.setInvertSelection(true);
	
	for(Map.Entry<String, Integer> entry : nominalSplitter.entrySet()) {
	    removeFilter.setAttributeIndex("" + (splitterAttributeID+1));
	    removeFilter.setNominalIndices("" + (nodeTrainingData.attribute(splitterAttributeID).indexOfValue(entry.getKey())+1));
	    removeFilter.setInputFormat(nodeTrainingData);

	    Instances subData = Filter.useFilter(nodeTrainingData, removeFilter);
	    subData.deleteAttributeAt(splitterAttributeID);
	    subTrees.set(entry.getValue(), new ID3DecisionTree(this, subData));	    
	}
    }
    
    public final ArrayList<ID3DecisionTree> getSubTrees() throws Exception {
	return subTrees;
    }
    public final Instances getNodeData() throws Exception {
	return nodeTrainingData;
    }
    
    /**
     * If this node is a leaf or numeric node, will return an empty array,
     * @param data The instance to be classified
     * @return The weight distribution array of the node's subtree according to the instance
     */
    public double[] getSubTreeDistribution(Instance data) {
	double[] res = new double[subTrees.size()];
	if(splitterAttributeID != -1) {
	    Attribute splitterAttribute = nodeTrainingData.attribute(splitterAttributeID);
	    Enumeration<Attribute> attributes = data.enumerateAttributes();
	    
	    while(attributes.hasMoreElements()) {
		Attribute nextAttributes = attributes.nextElement();
		if(nextAttributes.name() == null ? splitterAttribute.name() == null : nextAttributes.name().equals(splitterAttribute.name())) {
		    if(!data.isMissing(nextAttributes)) {
			String val = data.stringValue(nextAttributes); 
			res[nominalSplitter.get(val)] = 1.0;
		    }
		}
	    }
	}
	return res;
    }
    
    /**  
     * @return The weight distribution array of this node's class attributes value
     * @throws java.lang.Exception */
    public final double[] getClassDistribution() throws Exception {
	int[] counts;
	double[] weights;
	
	Instances data;
	if(!nodeTrainingData.isEmpty()) {
	    data = nodeTrainingData;
	} else {
	    data = parent.getNodeData();
	}
	
	counts = data.attributeStats(data.classIndex()).nominalCounts;
	weights = new double[counts.length];
	for(int i=0; i<counts.length; i++) {
	    weights[i] = ((double)counts[i])/((double)data.size());
	}
	
	return weights;
    }
    
    /** @return Whether or not this node is a leaf node */
    public final boolean isLeaf() {
	return isLeaf;
    }
    
    @Override
    public String toString() {
	try {
	    return toString(0);
	} catch (Exception ex) {
	    Logger.getLogger(ID3DecisionTree.class.getName()).log(Level.SEVERE, null, ex);
	    return "";
	}
    }
    
    public String toString(int level) throws Exception {
	StringBuilder sb = new StringBuilder();
	if(!isLeaf()) {
	    sb.append(tabLevel(level));
	    sb.append(String.format("Separator = %s\n", nodeTrainingData.attribute(splitterAttributeID)));
	    for(Map.Entry<String, Integer> entry : nominalSplitter.entrySet()) {
		sb.append(tabLevel(level));
		sb.append(String.format("Case %s,\n", entry.getKey()));
		sb.append(subTrees.get(entry.getValue()).toString(level + 1));
	    }
	} else {
	    sb.append(tabLevel(level));
	    sb.append("Leaf Node: ");
	    sb.append(Arrays.toString(getClassDistribution()));
	    sb.append("\n");
	}
	return sb.toString();
    }
    private String tabLevel(int level) {
	StringBuilder sb = new StringBuilder();
	for(int i=0; i<level; i++) {
	    sb.append("   ");
	}
	return sb.toString();
    }
}
