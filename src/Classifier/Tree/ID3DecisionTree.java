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
    
    /** The parent of this node */
    protected final ID3DecisionTree parent;
    
    /** The training data used to make this node (and its subtrees) */
    protected final Instances nodeTrainingData;
    
    protected boolean isLeaf;
    
    /** List of subtree of this node indexed by its ID */
    protected ArrayList<ID3DecisionTree> subTrees;
    
    /** The attribute ID used to separate the data at this node */
    protected Attribute splitterAttribute;
    
    /** The splitter that contains mapping of value to the subtree ID
     * If splitterAttribute is Nominal, contain direct mapping of value to ID
     * If Numeric, contain list of upperBound value and the mapping to ID (up to Double.MAX_VALUE)
     * ex. 13 -> 1, 17 -> 2, Inf -> 3, then 5 is mapped to 1, 14 to 2, and 19 to 3
     */
    protected TreeMap<Double, Integer> splitterMap;
    
    
    /** Make a new tree node with the supplied training data */
    public ID3DecisionTree(ID3DecisionTree parentNode, Instances subData) {
	isLeaf = true;
	splitterAttribute = null;
	nodeTrainingData = subData;
	splitterMap = new TreeMap<>();
	subTrees = new ArrayList<>();
	parent = parentNode;
    }
    
    /**
     * Will setup the splitterAttribute and initialize the subtree of this node 
     * by using the attID as the splitterAttribute attribute
     * @param attID The attribute ID this node will use to split the data
     * @throws java.lang.Exception
     */
    public final void SetNominalSplitter(int attID) throws Exception {
	splitterMap.clear();
	subTrees.clear();
	
	isLeaf = false;
	splitterAttribute = nodeTrainingData.attribute(attID);
	Enumeration<Object> values = splitterAttribute.enumerateValues();
	int i = 0;

	while(values.hasMoreElements()) {
	    String value = values.nextElement().toString();
	    splitterMap.put((double)splitterAttribute.indexOfValue(value), i);
	    subTrees.add(null);
	    i++;
	}
	
	// Setup subtrees according to the splitter
	setupSubTrees();
    }
    
    /**
     * Setup the subtrees according to the defined splitter
     * @throws Exception
     */
    protected void setupSubTrees() throws Exception {
	RemoveWithValues removeFilter = new RemoveWithValues();
	removeFilter.setInvertSelection(true);
	removeFilter.setMatchMissingValues(false);
	
	for(Map.Entry<Double, Integer> entry : splitterMap.entrySet()) {
	    removeFilter.setAttributeIndex("" + (splitterAttribute.index()+1));
	    removeFilter.setNominalIndices("" + (entry.getKey().intValue()+1));
	    removeFilter.setInputFormat(nodeTrainingData);

	    Instances subData = Filter.useFilter(nodeTrainingData, removeFilter);
	    subData.deleteAttributeAt(splitterAttribute.index());
	    subTrees.set(entry.getValue(), new ID3DecisionTree(this, subData));	    
	}
    }
    
    public final ArrayList<ID3DecisionTree> getSubTrees() {
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
	if(splitterAttribute != null) {
	    Enumeration<Attribute> attributes = data.enumerateAttributes();
	    
	    while(attributes.hasMoreElements()) {
		Attribute nextAttributes = attributes.nextElement();
		if(nextAttributes.name().equals(splitterAttribute.name())) {
		    if(!data.isMissing(nextAttributes)) {
			String val = data.stringValue(nextAttributes); 
			res[splitterMap.get((double)splitterAttribute.indexOfValue(val))] = 1.0;
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
	    sb.append(String.format("Separator = %s\n", nodeTrainingData.attribute(splitterAttribute.index())));
	    for(Map.Entry<Double, Integer> entry : splitterMap.entrySet()) {
		sb.append(tabLevel(level));
		sb.append(String.format("Case %s,\n", nodeTrainingData.attribute(splitterAttribute.index()).value(entry.getKey().intValue())));
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
    protected static final String tabLevel(int level) {
	StringBuilder sb = new StringBuilder();
	for(int i=0; i<level; i++) {
	    sb.append("   ");
	}
	return sb.toString();
    }
    
    protected void appendData(Instances data) {
	nodeTrainingData.addAll(data.subList(0, data.size()));
    }
}
