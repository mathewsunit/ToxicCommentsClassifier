package toxic

import java.io._

import org.apache.hadoop.fs.{FSDataOutputStream, FileSystem, Path}
import org.apache.spark.ml._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object MainObj {

  def main(args: Array[String]): Unit = {
    if(args.length < 3) {
      println("Expecting three parameters in the order <Training file>" +
        " <Test file> <Output Directory>")
      return
    }
    val sc = SparkSession.builder().appName("ToxicCommentsClassifier")
      .config("spark.master", "local").getOrCreate()
    sc.sparkContext.setLogLevel("ERROR")
    val trainSet: DataFrame = readInput(sc, args(0).toString, trainSchema)
    val testSet: DataFrame = readInput(sc, args{1}.toString, null)
    println("Number of rows in training data: " + trainSet.count())
    println("Number of rows in test data: " + testSet.count())
    println("Splitting training data to 75% training and 25% testing data")
    val Array(trainData, testData) = trainSet.randomSplit(Array(0.9, 0.1), seed = 12345)
    buildPredictUsingNB(sc, trainData, testData, testSet, args(2).toString)
    //buildPredictUsingMLP(trainData, testData, testSet)
  }

  def trueValue = true

  def trainSchema = StructType(Array(
    StructField("id", StringType, trueValue),
    StructField("comment_text", StringType, trueValue),
    StructField("toxic", DoubleType, trueValue),
    StructField("severe_toxic", DoubleType, trueValue),
    StructField("obscene", DoubleType, trueValue),
    StructField("threat", DoubleType, trueValue),
    StructField("insult", DoubleType, trueValue),
    StructField("identity_hate", DoubleType, trueValue)))

  def readInput(sc:SparkSession, path: String, schema: StructType): DataFrame = {
    sc.read.format("csv").option("header", "true").option("quote", "\"")
      .option("escape", "\"").option("multiline", "true").schema(schema)
      .load(path)
  }

  def getNBModel(dataFrame: DataFrame): PipelineModel = {
    val tokenizer = new Tokenizer().setInputCol("comment_text").setOutputCol("words")
    val stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
    val hashingTF = new HashingTF().setInputCol(tokenizer.getOutputCol).setOutputCol("features")
    val nb = new NaiveBayes()
    val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, hashingTF, nb))
    pipeline.fit(dataFrame)
  }

  def getMLPModel(dataFrame: DataFrame): PipelineModel = {
    val tokenizer = new Tokenizer().setInputCol("comment_text").setOutputCol("words")
    val stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
    val hashingTF = new HashingTF().setInputCol(tokenizer.getOutputCol).setOutputCol("features")
    val layers = Array[Int](10, 5, 5, 2)
    val mlp = new MultilayerPerceptronClassifier().setLayers(layers).setMaxIter(100)
    val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, hashingTF, mlp))
    pipeline.fit(dataFrame)
  }

  def getPredictionForModel(pModel: PipelineModel, dataFrame: DataFrame, name: String): DataFrame = {
    pModel.transform(dataFrame).select("id", "prediction")
      .withColumnRenamed("prediction", name)
  }

  def joinPredictions(parentData: DataFrame, toxicPredictions: DataFrame, severeToxicPredictions: DataFrame,
                      obscenePredictions: DataFrame, threatPredictions: DataFrame,
                      insultPredictions: DataFrame, identityHatePredictions: DataFrame): DataFrame = {
    parentData.join(toxicPredictions, "id").join(severeToxicPredictions, "id")
      .join(obscenePredictions, "id").join(threatPredictions, "id")
      .join(insultPredictions, "id").join(identityHatePredictions, "id")
  }

  def buildPredictUsingNB(sc: SparkSession, trainData: DataFrame, testData: DataFrame, testSet: DataFrame,
                          destPath: String): Unit = {
    println("Building Naive Bayes model")
    val toxicNBModel = getNBModel(trainData.withColumnRenamed("toxic", "label"))
    val toxicPredictions = getPredictionForModel(toxicNBModel, testData, "toxic_predicted")
    val severeToxicNBModel = getNBModel(trainData.withColumnRenamed("severe_toxic",
      "label"))
    val severeToxicPredictions = getPredictionForModel(severeToxicNBModel, testData,
      "severe_toxic_predicted")
    val obsceneNBModel = getNBModel(trainData.withColumnRenamed("obscene", "label"))
    val obscenePredictions = getPredictionForModel(obsceneNBModel, testData, "obscene_predicted")
    val threatNBModel = getNBModel(trainData.withColumnRenamed("threat", "label"))
    val threatPredictions = getPredictionForModel(threatNBModel, testData, "threat_predicted")
    val insultNBModel = getNBModel(trainData.withColumnRenamed("insult", "label"))
    val insultPredictions = getPredictionForModel(insultNBModel, testData, "insult_predicted")
    val identityHateNBModel = getNBModel(trainData.withColumnRenamed("identity_hate",
      "label"))
    val identityHatePredictions = getPredictionForModel(identityHateNBModel, testData,
      "identity_hate_predicted")
    println("Predicting using Naive Bayes model")
    val nbPredicted = joinPredictions(testData, toxicPredictions, severeToxicPredictions,
      obscenePredictions, threatPredictions, insultPredictions, identityHatePredictions)

    val fs = FileSystem.get(sc.sparkContext.hadoopConfiguration)
    val path: Path = new Path(destPath+"/NBAccuracy.txt")
    if (fs.exists(path)) {
      fs.delete(path, true)
    }
    val dataOutputStream: FSDataOutputStream = fs.create(path)
    val bw: BufferedWriter = new BufferedWriter(new OutputStreamWriter(dataOutputStream, "UTF-8"))

    printAccuracy(nbPredicted, "toxic", bw)
    printAccuracy(nbPredicted, "severe_toxic", bw)
    printAccuracy(nbPredicted, "obscene", bw)
    printAccuracy(nbPredicted, "threat", bw)
    printAccuracy(nbPredicted, "insult", bw)
    printAccuracy(nbPredicted, "identity_hate", bw)
    //please try the below code. We need to print the confusion matrix for this using MulticlassMetrics
    bw.write(SparkUtils.evaluateModel(nbPredicted.select("toxic", "toxic_predicted").rdd.map{ case Row(prediction: Double, label: Double) => (prediction, label) },"Toxic"))
    bw.write(SparkUtils.evaluateModel(nbPredicted.select("severe_toxic", "severe_toxic_predicted").rdd.map{ case Row(prediction: Double, label: Double) => (prediction, label) },"Severe Toxic"))
    bw.write(SparkUtils.evaluateModel(nbPredicted.select("obscene", "obscene_predicted").rdd.map{ case Row(prediction: Double, label: Double) => (prediction, label) },"Obscene"))
    bw.write(SparkUtils.evaluateModel(nbPredicted.select("threat", "threat_predicted").rdd.map{ case Row(prediction: Double, label: Double) => (prediction, label) },"Threat"))
    bw.write(SparkUtils.evaluateModel(nbPredicted.select("insult", "insult_predicted").rdd.map{ case Row(prediction: Double, label: Double) => (prediction, label) },"Insult"))
    bw.write(SparkUtils.evaluateModel(nbPredicted.select("identity_hate", "identity_hate_predicted").rdd.map{ case Row(prediction: Double, label: Double) => (prediction, label) },"Identity Hate"))
\    bw.close()
    /*val metrics = new MulticlassMetrics(nbPredicted.select("toxic", "toxic_predicted").rdd
    .flatMap(row => row.getDouble(0))
    println(metrics.accuracy)
    println(metrics.confusionMatrix)*/
    nbPredicted.show()
  }

  def buildPredictUsingMLP(trainData: DataFrame, testData: DataFrame, testSet: DataFrame): Unit = {
    println("Building MLP model")
    val toxicMLPModel = getMLPModel(trainData.withColumnRenamed("toxic", "label"))
    val toxicPredictions = getPredictionForModel(toxicMLPModel, testData, "toxic_predicted")
    val severeToxicMLPModel = getMLPModel(trainData.withColumnRenamed("severe_toxic",
      "label"))
    val severeToxicPredictions = getPredictionForModel(severeToxicMLPModel, testData,
      "severe_toxic_predicted")
    val obsceneMLPModel = getMLPModel(trainData.withColumnRenamed("obscene",
      "label"))
    val obscenePredictions = getPredictionForModel(obsceneMLPModel, testData, "obscene_predicted")
    val threatMLPModel = getMLPModel(trainData.withColumnRenamed("threat", "label"))
    val threatPredictions = getPredictionForModel(threatMLPModel, testData, "threat_predicted")
    val insultMLPModel = getMLPModel(trainData.withColumnRenamed("insult", "label"))
    val insultPredictions = getPredictionForModel(insultMLPModel, testData, "insult_predicted")
    val identityHateMLPModel = getMLPModel(trainData.withColumnRenamed("identity_hate",
      "label"))
    val identityHatePredictions = getPredictionForModel(identityHateMLPModel, testData,
      "identity_hate_predicted")
    println("Predicting using MLP model")
    val mlpPredicted = joinPredictions(testData, toxicPredictions, severeToxicPredictions,
      obscenePredictions, threatPredictions, insultPredictions, identityHatePredictions)
    mlpPredicted.show()
  }

  def printAccuracy(predicted: DataFrame, toxicType: String, bw: BufferedWriter): Unit = {
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol(toxicType).setPredictionCol(
      toxicType+"_predicted").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predicted)
    println("Test set accuracy for " + toxicType + " = " + accuracy)
    bw.write("Test set accuracy for " + toxicType + " = " + accuracy + "\r\n")
  }
}