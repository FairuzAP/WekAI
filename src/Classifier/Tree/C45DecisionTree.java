/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Classifier.Tree;

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
    
    @Override
    public void SetNominalSplitter(int attID) throws Exception {
	super.SetNominalSplitter(attID);
    }
    
    @Override
    protected void setupSubTrees() throws Exception {
	super.setupSubTrees();
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
