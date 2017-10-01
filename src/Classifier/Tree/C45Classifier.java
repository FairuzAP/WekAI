/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Classifier.Tree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.instance.RemovePercentage ;
import javafx.util.Pair;
import weka.classifiers.AbstractClassifier;
import weka.filters.Filter;


/**
 *
 * @author USER
 */
public class C45Classifier extends ID3Classifier {
    
    // The data to be used for trimming / testing model
    protected Instances validationData;
    
    protected boolean isRule;
    
    protected class Constraint {
	String attName;
	double nominalValue;
	double lowerNumericBound;
	double upperNumericBound;
	
	public boolean isValid(Instance instance) {
	    boolean valid = true;
	    Enumeration<Attribute> attributes = instance.enumerateAttributes();
	    while(attributes.hasMoreElements()) {
		Attribute next = attributes.nextElement();
		if(next.name().equals(attName)) {

		    if(next.isNominal()) {
			if(instance.value(next) != nominalValue) {
			    valid = false;
			}
		    } else {
			if(instance.value(next) <= lowerNumericBound || instance.value(next) > upperNumericBound) {
			    valid = false;
			}
		    }
		    break;

		}
	    }
	    return valid;
	}
	
	@Override
	public String toString() {
	    try {
		if(nominalValue == -1) {
		    return lowerNumericBound + " > " + attName + " >= " + upperNumericBound;
		} else {
		    return attName + " = " + nominalValue;
		}
	    } catch (Exception ex) {
		Logger.getLogger(ID3DecisionTree.class.getName()).log(Level.SEVERE, null, ex);
		return "";
	    }
	}
    }
    protected class Rule {
	ArrayList<Constraint> constraint;
	double[] classDistribution;
	double accuracy;
	
	public boolean isValid(Instance instance) {
	    boolean valid = true;
	    for(Constraint c : constraint) {
		if(!c.isValid(instance)) {
		    valid = false;
		    break;
		}
	    }
	    return valid;
	}
	
	@Override
	public String toString() {
	    try {
		return constraint.toString() + " -> " + Arrays.toString(classDistribution) + ", a=" + accuracy + "\n";
	    } catch (Exception ex) {
		Logger.getLogger(ID3DecisionTree.class.getName()).log(Level.SEVERE, null, ex);
		return "";
	    }
	}
    }
    protected ArrayList<Rule> ruleSet;
    
    protected class RuleClassifier extends AbstractClassifier {
	protected Rule rule;
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
	    // Do Nothing
	}
	
	@Override
	public final double[] distributionForInstance(Instance instance) throws Exception {
	    if(!rule.isValid(instance)) {
		double[] res = new double[rule.classDistribution.length];
		for(int i=0; i<res.length; i++) {
		    res[i] = 0.0;
		}
		return res;
	    } else {
		return rule.classDistribution;
	    }
	    
	    
	}
    }
    protected class RuleComparator implements Comparator<Rule> {

	@Override
	public int compare(Rule o1, Rule o2) {
	    if(o1.accuracy > o2.accuracy) {
		return -1;
	    } else if(o1.accuracy < o2.accuracy) {
		return 1;
	    } else {
		return 0;
	    }
	}
    
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
	// Remove instance with missing class value
        data.randomize(new Random(System.currentTimeMillis()));
	data.deleteWithMissingClass();
	RemovePercentage removeFilt = new RemovePercentage ();

	removeFilt.setPercentage(10);
	removeFilt.setInvertSelection(true);
	removeFilt.setInputFormat(data);
	validationData = Filter.useFilter(data, removeFilt);
	
	removeFilt.setPercentage(10);
	removeFilt.setInvertSelection(false);
	removeFilt.setInputFormat(data);
	trainingData = Filter.useFilter(data, removeFilt);
	
	root = new C45DecisionTree(null, trainingData);
	setupTree(root);
	trimTree(validationData);
	
	System.out.println(root);
	System.out.println(ruleSet);
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
	ruleSet = new ArrayList<>();
	parseTreeToRules((C45DecisionTree) root, new ArrayList<>());
	
	for(int i=0; i< ruleSet.size(); i++) {
	    ruleSet.set(i, trimRule(ruleSet.get(i)));
	}
	
	RuleComparator c = new RuleComparator();
	ruleSet.sort(c);
	
	isRule = false;
	reducedErrorPrune((C45DecisionTree) root);
	isRule = true;
    }
    
    private void parseTreeToRules(C45DecisionTree node, ArrayList<Constraint> constraints) throws Exception {
	
	if(node.isLeaf) {
	    Rule r = new Rule();
	    r.constraint = constraints;
	    r.classDistribution = node.getClassDistribution();
	    
	    RuleClassifier classifier = new RuleClassifier();
	    classifier.rule = r;
	    Evaluation currEval = new Evaluation(trainingData);
	    currEval.evaluateModel(classifier, validationData);
	    r.accuracy = (currEval.correct()/currEval.numInstances());
	    
	    ruleSet.add(r);
	    
	} else {
	    if(node.splitterAttribute.isNominal()) {
		for(Map.Entry<Double, Integer> entry : node.splitterMap.entrySet()) {

		    ArrayList<Constraint> nextList = new ArrayList<>();
		    nextList.addAll(constraints);

		    Constraint nextConstraint = new Constraint();
		    nextConstraint.attName = node.splitterAttribute.name();
		    nextConstraint.nominalValue = entry.getKey();

		    nextList.add(nextConstraint);

		    parseTreeToRules((C45DecisionTree) node.subTrees.get(entry.getValue()), nextList);
		}
	    } else {
		double prevKey = Double.MIN_VALUE;
		for(Map.Entry<Double, Integer> entry : node.splitterMap.entrySet()) {

		    ArrayList<Constraint> nextList = new ArrayList<>();
		    nextList.addAll(constraints);

		    Constraint nextConstraint = new Constraint();
		    nextConstraint.attName = node.splitterAttribute.name();
		    nextConstraint.nominalValue = -1;
		    nextConstraint.lowerNumericBound = prevKey;
		    nextConstraint.upperNumericBound = entry.getKey();
		    prevKey = entry.getKey();

		    nextList.add(nextConstraint);

		    parseTreeToRules((C45DecisionTree) node.subTrees.get(entry.getValue()), nextList);
		}
	    }
	}	
    }
    private Rule trimRule(Rule rule) throws Exception {
	RuleClassifier classifier = new RuleClassifier();
	classifier.rule = rule;
	
	Evaluation currEval = new Evaluation(trainingData);
	currEval.evaluateModel(classifier, validationData);
	double prevError = (currEval.correct()/currEval.numInstances());
	
	if(rule.constraint.size() <= 1) {
	    for(Constraint c : rule.constraint) {
		Rule nextRule = new Rule();
		nextRule.constraint = new ArrayList<>(rule.constraint);
		nextRule.classDistribution = rule.classDistribution;
		nextRule.constraint.remove(c);
		classifier.rule = nextRule;

		currEval = new Evaluation(trainingData);
		currEval.evaluateModel(classifier, validationData);
		if(prevError < (currEval.correct()/currEval.numInstances())) {
		    return trimRule(nextRule);
		}
	    }
	}
	
	rule.accuracy = prevError;
	return rule;
    }
    
    private void reducedErrorPrune(C45DecisionTree node) throws Exception {
	if(!node.isLeaf()) {
	    for(int i=0; i<node.getSubTrees().size(); i++) {
		reducedErrorPrune((C45DecisionTree) node.getSubTrees().get(i));
	    }
	    
	    Evaluation old_eval = new Evaluation(trainingData);
	    old_eval.evaluateModel(this, validationData);
	    node.isLeaf = true;
	    Evaluation new_eval = new Evaluation(trainingData);
	    new_eval.evaluateModel(this, validationData);
	    
	    if(old_eval.rootMeanSquaredError() < new_eval.rootMeanSquaredError()) {
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
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
	if(isRule) {
	    for(Rule r : ruleSet) {
		if(r.isValid(instance)) {
		    return r.classDistribution;
		}
	    }
	    return ruleSet.get(0).classDistribution;
	} else {
	    return super.distributionForInstance(instance);
	}
    }
}