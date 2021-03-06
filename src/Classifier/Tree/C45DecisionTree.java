/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Classifier.Tree;

import static Classifier.Tree.ID3DecisionTree.tabLevel;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

/**
 *
 * @author USER
 */
public class C45DecisionTree extends ID3DecisionTree {
    
    public C45DecisionTree(ID3DecisionTree parentNode, Instances subData) {
	super(parentNode, subData);
    }
    
    /**
     * Will setup the splitterAttribute and initialize the subtree of this node 
     * by using the attID as the splitterAttribute attribute
     * @param attID The attribute ID this node will use to split the data
     * @param boundaries The boundary values that'll separate the attribute values
     * @throws java.lang.Exception
     */
    public final void SetNumericSplitter(int attID, double[] boundaries) throws Exception {
	splitterMap.clear();
	subTrees.clear();
	
	isLeaf = false;
	splitterAttribute = nodeTrainingData.attribute(attID);
	
	for(int i=0; i<boundaries.length; i++) {
	    splitterMap.put(boundaries[i], i);
	    subTrees.add(null);
	}
	splitterMap.put(Double.MAX_VALUE, boundaries.length);
	subTrees.add(null);
	
	// Setup subtrees according to the splitter
	setupSubTrees();
    }
    
    /**
     * Extend the ID3 method to add handler for continuous attributes, and
     * missing attributes value for the selected splitter
     * @throws Exception
     */
    @Override
    protected void setupSubTrees() throws Exception {
	if(splitterAttribute.isNumeric()) {
	    
	    RemoveWithValues removeFilter = new RemoveWithValues();
	    removeFilter.setMatchMissingValues(false);
	    Iterator<Double> keys = splitterMap.navigableKeySet().iterator();

	    double prev = Double.MIN_VALUE, curr;
	    while(keys.hasNext()) {
		curr = keys.next();
		removeFilter.setInvertSelection(true);
		removeFilter.setAttributeIndex("" + (splitterAttribute.index()+1));
		removeFilter.setSplitPoint(curr);
		removeFilter.setInputFormat(nodeTrainingData);
		
		Instances subData = Filter.useFilter(nodeTrainingData, removeFilter);
		if(prev != Double.MIN_VALUE) {
		    removeFilter.setInvertSelection(false);
		    removeFilter.setSplitPoint(prev);
		    removeFilter.setInputFormat(subData);
		    subData = Filter.useFilter(subData, removeFilter);
		}
		subData.deleteAttributeAt(splitterAttribute.index());

		subTrees.set(splitterMap.get(curr), new C45DecisionTree(this, subData));
		prev = curr;
	    }

	} else {
	    RemoveWithValues removeFilter = new RemoveWithValues();
	    removeFilter.setInvertSelection(true);
	    removeFilter.setMatchMissingValues(false);

	    for(Map.Entry<Double, Integer> entry : splitterMap.entrySet()) {
		removeFilter.setAttributeIndex("" + (splitterAttribute.index()+1));
		removeFilter.setNominalIndices("" + (entry.getKey().intValue()+1));
		removeFilter.setInputFormat(nodeTrainingData);

		Instances subData = Filter.useFilter(nodeTrainingData, removeFilter);
		subData.deleteAttributeAt(splitterAttribute.index());
		subTrees.set(entry.getValue(), new C45DecisionTree(this, subData));	    
	    }
	}
	
	// Add instances with missing attributes to the most common subtree
	RemoveWithValues removeFilter = new RemoveWithValues();
	removeFilter.setMatchMissingValues(true);
	removeFilter.setInvertSelection(false);
	removeFilter.setAttributeIndex("" + (splitterAttribute.index()+1));
	removeFilter.setSplitPoint(Double.MAX_VALUE);
	removeFilter.setInputFormat(nodeTrainingData);
	Instances missingData = Filter.useFilter(nodeTrainingData, removeFilter);
	if(!missingData.isEmpty()) {
	    subTrees.get(getMostCommonSubTreeIndex()).appendData(missingData);
	}
	
    }
    
    /**
     * Extend the previous method to add handler for continuous attributes
     * and for missing splitter attributes value
     * @param data The instance to be classified
     * @return The weight distribution array of the node's subtree according to the instance
     */
    @Override
    public double[] getSubTreeDistribution(Instance data) {
	double[] res = new double[subTrees.size()];
	if(splitterAttribute != null) {
	    Enumeration<Attribute> attributes = data.enumerateAttributes();

	    while(attributes.hasMoreElements()) {
		Attribute nextAttributes = attributes.nextElement();
		if(nextAttributes.name().equals(splitterAttribute.name())) {

		    if(!data.isMissing(nextAttributes)) {

			if(splitterAttribute.isNominal()) {
			    String val = data.stringValue(nextAttributes); 
			    Double key = (double)splitterAttribute.indexOfValue(val);
			    res[splitterMap.get(key)] = 1.0;
			} else {
			    Double key = splitterMap.ceilingKey(data.value(nextAttributes));
			    res[splitterMap.get(key)] = 1.0;
			}

		    } else {
			res[getMostCommonSubTreeIndex()] = 1.0;
		    }

		}
	    }
	}
	return res;
    }
    
    private int getMostCommonSubTreeIndex() {
	int maxSize = Integer.MIN_VALUE;
	int res = 0;
	for(int i=0; i<subTrees.size(); i++) {
	    int currSize = 0;
	    
	    try {
		currSize = subTrees.get(i).getNodeData().size();
	    } catch (Exception ex) {
		Logger.getLogger(C45DecisionTree.class.getName()).log(Level.SEVERE, null, ex);
	    }
	    
	    if(currSize > maxSize) {
		maxSize = currSize;
		res = i;
	    }
	}
	return res;
    }
    
    @Override
    public String toString(int level) throws Exception {
	StringBuilder sb = new StringBuilder();
	if(!isLeaf()) {
	    sb.append(tabLevel(level));
	    sb.append(String.format("Separator = %s\n", nodeTrainingData.attribute(splitterAttribute.index())));
	    for(Map.Entry<Double, Integer> entry : splitterMap.entrySet()) {
		sb.append(tabLevel(level));
		if(splitterAttribute.isNominal()) {
		    sb.append(String.format("Case %s,\n", nodeTrainingData.attribute(splitterAttribute.index()).value(entry.getKey().intValue())));
		} else {
		    sb.append(String.format("Case ... > x >= %s,\n", entry.getKey()));
		}
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
}
