/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekai;

import java.text.ParseException;
import java.util.Enumeration;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 *
 * @author USER
 */
public class WekaHandler {
    
    /**
     * The training data 
     */
    public Instances Data;
    /**
     * The most recent trained / untrained Classifier for this data
     */
    public Classifier Model;
    /**
     * The most recent evaluation of a training
     */
    public Evaluation eval;
    /**
     * The classified from training data 
     */
    public Instances Result;
    
    /**
     * Load data from the usual file type
     * @param filepath the file path of the loaded file
     * @return TRUE if data was read succesfully, FALSE if otherwise
     */
    public boolean readData(String filepath) {
	try {
	    Data = ConverterUtils.DataSource.read(filepath);
	    Data.setClassIndex(0);
	    Result = new Instances(Data,0);
	    Result.setClassIndex(Result.numAttributes()-1);
	} catch (Exception ex) {
	    System.out.println(ex);
	    return false;
	}
	return true;
    }
    
    /**
     * Shuffle the order of the data
     */
    public void randomize() {
	Random RNGesus = Data.getRandomNumberGenerator(System.currentTimeMillis());
	Data.randomize(RNGesus);
    }
    
    
    /**
     * Discretize data according to the parameters
     * @param attributes The id's of the attributes to be discretized, 
     * @param binnumber the desired number of bins
     * @param intervalweight the desired weight of instances per interval
     * @return true if all goes well
     */
    public boolean discretize(int[] attributes, int binnumber, int intervalweight) {
	Discretize d = new Discretize();
	d.setAttributeIndicesArray(attributes);
	d.setBins(binnumber);
	d.setDesiredWeightOfInstancesPerInterval(intervalweight);
	return filterize(d);
    }
    /**
     * Convert the attributes in Data from numeric to nominal
     * @param attributes The id's of the attributes to be converted to nominal,
     * if NULL all will be discretized
     * @return true if all goes well
     */
    public boolean numericToNominal(int[] attributes) {
	NumericToNominal n = new NumericToNominal();
	n.setAttributeIndicesArray(attributes);
	return filterize(n);
    }
    
    /**
     * Apply a filter to the Data
     * @param f The filter to be applied
     * @return true if all goes well
     */
    public boolean filterize(Filter f) {
	try {
	    f.setInputFormat(Data);
	    Instances newData = Filter.useFilter(Data, f);
	    Data = newData;
	} catch (Exception ex) {
		
	    return false;
	}
	return true;
    }
    
    
    /**
     * Set the classifier to be used in learning
     * @param d The new classifier
     */
    public void setClassifier(Classifier d) {
	Model = d;
    }
	 
    /**
     * Learning dataset, 10-fold cross-validation; if no model is provided
     * then use NaiveBayes model by default
     */
    public void tenFoldCrossValidation(){
	try {
	    
	    //Build classifier
	    if(Model==null) Model = new NaiveBayes();
    	    long startTime = System.currentTimeMillis();
	    Model.buildClassifier(Data);
	    long endTime   = System.currentTimeMillis();

	    //Evaluate classifier
	    int folds = 10;
	    eval = new Evaluation(Data);
	    eval.crossValidateModel(Model, Data, folds, new Random(1));

	    //Print statistic result
	    printResult(0,startTime,endTime,folds,0);
	} catch(Exception ex) {
	    System.out.println(ex);
	}
    }
    /**
     * Learning dataset, full-training; if no model is provided then use 
     * NaiveBayes model by default
     */
    public void fullTraining(){
	try {
	    
	    //Train classifier
	    if(Model==null) Model = new NaiveBayes();
	    long startTime = System.currentTimeMillis();
	    Model.buildClassifier(Data);
	    long endTime   = System.currentTimeMillis();
	    
	    //Evaluate classifier
	    eval = new Evaluation(Data);
	    long startT = System.currentTimeMillis();
	    eval.evaluateModel(Model, Data);
	    long endT   = System.currentTimeMillis();
	    
	    //Print statistics
	    printResult(1,startTime,endTime,startT,endT);
	    
	} catch(Exception ex) {
	    System.out.println(ex);
	}
    }
    
    
    /**
     * Print result learning dataset
     * @param scheme 0=10-fold cross-validation, 1=full-training
     * @param startTime start time of building model
     * @param endTime end time of building model
     * @param prop1 another properties from each scheme
     * @param prop2 another properties from each scheme
     */
    public void printResult(int scheme,long startTime,long endTime,long prop1,long prop2) {
	try {
	    // Print result
	    System.out.println("=== Run information ===\n");
	    System.out.println("Scheme:       " + Model.getClass().getName());
	    System.out.println("Relation:     " + Data.relationName());
	    System.out.println("Instances:    " + Data.numInstances());
	    System.out.println("Attributes:   " + Data.numAttributes());
	    for(int i=0;i<Data.numAttributes();i++){
		System.out.println("              " + Data.attribute(i).name());
	    }
	    
	    if (scheme==0){
		System.out.println("Test mode:    "+ prop1 +"-fold cross-validation");
	    }else{
		System.out.println("Test mode:    evaluate on training data");
	    }

	    System.out.println("\n=== Classifier model (full training set) ===\n");
	    System.out.println(Model.toString());

	    System.out.println("\nTime taken to build model: " + String.format("%.2f",(endTime-startTime)/1000.0) + " seconds\n");

	    if (scheme==0){
		System.out.println("=== Stratified cross-validation ===");
	    }else{
		System.out.println("=== Evaluation on training set ===\n");
		System.out.println("Time taken to test model on training data: "+ String.format("%.2f",(prop2-prop1)/1000.0) +" seconds\n");
	    }

	    System.out.println(eval.toSummaryString("=== Summary ===\n",false));
	    System.out.println(eval.toClassDetailsString());
	    System.out.println(eval.toMatrixString());
	}catch (Exception ex){
	    System.out.println(ex);
	}

    }
    public void printEvalResult() {
	try {
	    eval = new Evaluation(Data);
	    eval.evaluateModel(Model, Data);
	    
	    // Print result
	    System.out.println("=== Run information ===\n");
	    System.out.println("Scheme:       " + Model.getClass().getName());
	    System.out.println("Relation:     " + Data.relationName());
	    System.out.println("Instances:    " + Data.numInstances());
	    System.out.println("Attributes:   " + Data.numAttributes());
	    for(int i=0;i<Data.numAttributes();i++){
		System.out.println("              " + Data.attribute(i).name());
	    }

	    System.out.println("\n=== Classifier model (full training set) ===\n");
	    System.out.println(Model.toString());

	    System.out.println(eval.toSummaryString("=== Summary ===\n",false));
	    System.out.println(eval.toClassDetailsString());
	    System.out.println(eval.toMatrixString());
	}catch (Exception ex){
	    System.out.println(ex);
	}
    }
    
    /**
     * Save Model to external file
     * @param filepath path tot the external file
     */
    public void saveModel(String filepath){
	try{
	    SerializationHelper.write(filepath, Model);
	}catch(Exception ex){
	    System.out.println(ex);
	}
    }
    /**
     * Read Model from external file
     * @param filepath path tot the external file
     */
    public void readModel(String filepath){
	try{
	    Model = (Classifier) SerializationHelper.read(filepath);
	}catch(Exception ex){
	    System.out.println(ex);
	}
    }
    
    
    /**
     * Create an instance of the 'Data' Instances from cmd input's and append 
     * it to Result's instances
     * @throws java.lang.Exception if an unrecognized input is read
     */
    public void getInstance() throws Exception {
	
	double values[] = new double[Data.numAttributes()];
	Scanner in = new Scanner(System.in);
	for(int i=0; i < Data.numAttributes()-1; i++) {
	    
	    Attribute att = Data.attribute(i);
	    String attype = Attribute.typeToString(att);
	    System.out.format("Please give '%s' %s attribute value\n", att.name(), attype);
	    
	    if(att.isNominal()) {
		Enumeration<Object> e = att.enumerateValues();
		System.out.print("Possible value = (");
		while(e.hasMoreElements()) System.out.format("%s, ",e.nextElement().toString());
		System.out.println("\b\b)");
	    }
	    if(att.isDate()) {
		System.out.format("The format is '%s'\n", att.getDateFormat());
	    }
	    
	    String input = in.next();
	    values[i] = parseAttInput(att, input);
	}
	
	Instance res = new DenseInstance(1.0,values);
	Result.add(res);
    }
    private double parseAttInput(Attribute att, String input) throws Exception {
	String attype = Attribute.typeToString(att);
	double res = 0;
	
	switch (attype) {
	    case "numeric":
		res = Double.parseDouble(input);
		break;
	    case "date":
		try {
		    res = att.parseDate(input);
		} catch (ParseException ex) {
		    throw new Exception("ERROR WHEN PARSING DATE");
		}
		break;
	    case "nominal":
		res = att.indexOfValue(input);
		break;
	    case "string":
		res = att.addStringValue(input);
		break;
	    case "relational":
		// CURRENTLY NOT SUPPORTED
		throw new Exception("RELATIONAL TYPE NOT SUPPORTED");
	    default:
		throw new Exception("UNRECOGNIZED ATT TYPE");
	}
	
	return res;
    }
    
    /**
     * Set the class value for all instance in result according to classifier
     * @return true if all goes well
     */
    public boolean classifyInstance() {
	try {
	    
	    for (int i = 0; i < Result.numInstances(); i++) {
		double clsLabel = Model.classifyInstance(Result.instance(i));
		Result.instance(i).setClassValue(clsLabel);
	    }
	    
	} catch (Exception ex) {
	    return false;
	}
	return true;
    }
    
}
