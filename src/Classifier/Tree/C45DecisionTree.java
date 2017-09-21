/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Classifier.Tree;

import java.util.Enumeration;
import java.util.TreeMap;
import weka.core.Instance;
import weka.core.Instances;

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
     * @throws java.lang.Exception
     */
    public final void SetNumericSplitter(int attID) throws Exception {
	splitterMap.clear();
	subTrees.clear();
	
	isLeaf = false;
	splitterAttribute = nodeTrainingData.attribute(attID);;

	// TODO: Here
	
	setupSubTrees();
    }
    
    
    @Override
    protected void setupSubTrees() throws Exception {
	if(splitterAttribute.isNominal()) {
	    super.setupSubTrees();
	} else {
	    
	    // TODO: Here
	    
	}
    }
    
    /**
     * If this node is a leaf or numeric node, will return an empty array,
     * @param data The instance to be classified
     * @return The weight distribution array of the node's subtree according to the instance
     */
    @Override
    public double[] getSubTreeDistribution(Instance data) {
	return super.getSubTreeDistribution(data);
    }
    
    
}
