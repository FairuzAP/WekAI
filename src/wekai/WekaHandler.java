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
import java.util.logging.Level;
import java.util.logging.Logger;
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
import weka.filters.unsupervised.instance.Resample;

/**
 *
 * @author USER
 */
public class WekaHandler {
    
    /**
     * The training data 
     */
    public Instances TrainingData;
    /**
     * The validation data 
     */
    public Instances ValidationData;
    
    /**
     * The most recent trained / untrained Classifier for this data
     */
    public Classifier Model;
    /**
     * The most recent evaluation of a training
     */
    private Evaluation eval;
    
    /**
     * The classified from training data 
     */
    private Instances UnclassifiedData;
    
    /**
     * Load data from the usual file type
     * @param filepath the file path of the loaded file
     * @return TRUE if data was read succesfully, FALSE if otherwise
     */
    public boolean readTrainingData(String filepath) {
	try {
	    TrainingData = ConverterUtils.DataSource.read(filepath);
	    TrainingData.setClassIndex(TrainingData.numAttributes()-1);
	    ValidationData = TrainingData;
	    
	    UnclassifiedData = new Instances(TrainingData,0);
	    UnclassifiedData.setClassIndex(UnclassifiedData.numAttributes()-1);
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
	Random RNGesus = TrainingData.getRandomNumberGenerator(System.currentTimeMillis());
	TrainingData.randomize(RNGesus);
    }
    
    /**
     * Apply a filter to the TrainingData
     * @param f The filter to be applied
     * @return true if all goes well
     */
    public boolean filterize(Filter f) {
	try {
	    f.setInputFormat(TrainingData);
	    Instances newData = Filter.useFilter(TrainingData, f);
	    TrainingData = newData;
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
     * @param train Whether or not to build the classifier, or just validate it
     */
    public void tenFoldCrossTraining(boolean train){
	try {
	    
	    //Build classifier
	    if(Model==null) Model = new NaiveBayes();
    	    long startTime = System.currentTimeMillis();
	    if(train) {
		Model.buildClassifier(TrainingData);
	    }
	    long endTime   = System.currentTimeMillis();

	    //Evaluate classifier
	    int folds = 10;
	    eval = new Evaluation(TrainingData);
	    eval.crossValidateModel(Model, ValidationData, folds, new Random(1));

	    //Print statistic result
	    printResult(startTime,endTime);
	} catch(Exception ex) {
	    System.out.println(ex);
	}
    }
    /**
     * Learning dataset, full-training; if no model is provided then use 
     * NaiveBayes model by default
     * @param train Whether or not to build the classifier, or just validate it
     */
    public void fullTraining(boolean train){
	try {
	    
	    //Train classifier
	    if(Model==null) Model = new NaiveBayes();
	    long startTime = System.currentTimeMillis();
	    if(train) {
		Model.buildClassifier(TrainingData);
	    }
	    long endTime   = System.currentTimeMillis();
	    
	    //Evaluate classifier
	    eval = new Evaluation(TrainingData);
	    eval.evaluateModel(Model, ValidationData);
	    
	    //Print statistics
	    printResult(startTime,endTime);
	    
	} catch(Exception ex) {
	    Logger.getLogger(WekaHandler.class.getName()).log(Level.SEVERE, null, ex);
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
    public void printResult(long startTime,long endTime) {
	try {
	    // Print result
	    System.out.println("=== Run information ===\n");
	    System.out.println("Scheme:       " + Model.getClass().getName());
	    System.out.println("Relation:     " + TrainingData.relationName());
	    System.out.println("Instances:    " + TrainingData.numInstances());
	    System.out.println("Attributes:   " + TrainingData.numAttributes());
	    for(int i=0;i<TrainingData.numAttributes();i++){
		System.out.println("              " + TrainingData.attribute(i).name());
	    }

	    System.out.println("\n=== Classifier model (full training set) ===\n");
	    System.out.println(Model.toString());
	    System.out.println("\nTime taken to build model: " + String.format("%.2f",(endTime-startTime)/1000.0) + " seconds\n");

	    System.out.println(eval.toSummaryString("=== Evaluation Summary ===\n",false));
	    System.out.println(eval.toClassDetailsString());
	    System.out.println(eval.toMatrixString());
	}catch (Exception ex){
	    System.out.println(ex);
	}

    }
    public void printEvalResult() {
	try {
	    eval = new Evaluation(TrainingData);
	    eval.evaluateModel(Model, TrainingData);
	    
	    // Print result
	    System.out.println("=== Run information ===\n");
	    System.out.println("Scheme:       " + Model.getClass().getName());
	    System.out.println("Relation:     " + TrainingData.relationName());
	    System.out.println("Instances:    " + TrainingData.numInstances());
	    System.out.println("Attributes:   " + TrainingData.numAttributes());
	    for(int i=0;i<TrainingData.numAttributes();i++){
		System.out.println("              " + TrainingData.attribute(i).name());
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
     * Create an instance of the 'TrainingData' Instances from cmd input's and 
 append it to UnclassifiedData's instances
     * @throws java.lang.Exception if an unrecognized input is read
     */
    public void getInstance() throws Exception {
	
	double values[] = new double[TrainingData.numAttributes()];
	Scanner in = new Scanner(System.in);
	for(int i=0; i < TrainingData.numAttributes()-1; i++) {
	    
	    Attribute att = TrainingData.attribute(i);
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
	UnclassifiedData.add(res);
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
	    
	    for (int i = 0; i < UnclassifiedData.numInstances(); i++) {
		double clsLabel = Model.classifyInstance(UnclassifiedData.instance(i));
		UnclassifiedData.instance(i).setClassValue(clsLabel);
	    }
	    
	} catch (Exception ex) {
	    return false;
	}
	return true;
    }
}