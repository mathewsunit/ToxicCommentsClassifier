package toxic

import java.io._

import org.apache.hadoop.fs.{FSDataOutputStream, FileSystem, Path}
import org.apache.spark.ml._
import org.apache.spark.ml.classification._
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
    val Array(trainData, testData) = trainSet.randomSplit(Array(0.75, 0.25), seed = 12345)
//    buildPredictUsingNB(sc, trainData, testData, testSet, args(2).toString)
    buildPredictUsingMLP(sc, trainData, testData, testSet,  args(2).toString)
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

  def getNBPipeline: Pipeline = {
    val tokenizer = new Tokenizer().setInputCol("comment_text").setOutputCol("words")
    val stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
    val hashingTF = new HashingTF().setInputCol(tokenizer.getOutputCol).setOutputCol("features")
    val nb = new NaiveBayes()
    val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, hashingTF, nb))
    pipeline
  }

  def getMLPPipeLine: Pipeline = {
    val tokenizer = new Tokenizer().setInputCol("comment_text").setOutputCol("words")
    val stopWordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
    val hashingTF = new HashingTF().setInputCol(stopWordsRemover.getOutputCol).setOutputCol("features").setNumFeatures(20)
    val layers = Array[Int](hashingTF.getNumFeatures, 10, 5, 2)
    val mlp = new MultilayerPerceptronClassifier().setLayers(layers).setMaxIter(100)
    val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, hashingTF, mlp))
    pipeline
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

    // Initialize Buffered Streams
    val fs = FileSystem.get(sc.sparkContext.hadoopConfiguration)
    val path: Path = new Path(destPath+"/NBAccuracy.txt")
    if (fs.exists(path)) {
      fs.delete(path, true)
    }
    val dataOutputStream: FSDataOutputStream = fs.create(path)
    val bw: BufferedWriter = new BufferedWriter(new OutputStreamWriter(dataOutputStream, "UTF-8"))

    // Generate Pipeline
    val pipeline = getNBPipeline

    // Generate Predictions
    val toxicPredictions = getPredictions(trainData, testData, getMLPPipeLine, "toxic")
    val severeToxicPredictions = getPredictions(trainData, testData, getMLPPipeLine,  "severe_toxic")
    val obscenePredictions = getPredictions(trainData, testData, getMLPPipeLine, "obscene")
    val threatPredictions = getPredictions(trainData, testData, getMLPPipeLine, "threat")
    val insultPredictions = getPredictions(trainData, testData, getMLPPipeLine, "insult")
    val identityHatePredictions = getPredictions(trainData, testData, getMLPPipeLine, "identity_hate")

    //please try the below code. We need to print the confusion matrix for this using MulticlassMetrics
    bw.write(SparkUtils.evaluateModel(toxicPredictions.rdd.map{ case Row(prediction: Double, label: Double) => (label, prediction) },"Toxic"))
    bw.write(SparkUtils.evaluateModel(severeToxicPredictions.rdd.map{ case Row(prediction: Double, label: Double) => (label, prediction) },"Severe Toxic"))
    bw.write(SparkUtils.evaluateModel(obscenePredictions.rdd.map{ case Row(prediction: Double, label: Double) => (label, prediction) },"Obscene"))
    bw.write(SparkUtils.evaluateModel(threatPredictions.rdd.map{ case Row(prediction: Double, label: Double) => (label, prediction) },"Threat"))
    bw.write(SparkUtils.evaluateModel(insultPredictions.rdd.map{ case Row(prediction: Double, label: Double) => (label, prediction) },"Insult"))
    bw.write(SparkUtils.evaluateModel(identityHatePredictions.rdd.map{ case Row(prediction: Double, label: Double) => (label, prediction) },"Identity Hate"))
    bw.close()
  }

  def getPredictions(trainData: DataFrame, testData: DataFrame, pipeline: Pipeline, labelName: String): DataFrame = {
    val prediction = pipeline.fit(trainData.withColumnRenamed(labelName, "label")).transform(testData).select(labelName, "prediction")
    prediction
  }

  def buildPredictUsingMLP(sc: SparkSession, trainData: DataFrame, testData: DataFrame, testSet: DataFrame,
                           destPath: String): Unit = {
    println("Building MLP model")

    // Initialize Buffered Streams
    val fs = FileSystem.get(sc.sparkContext.hadoopConfiguration)
    val path: Path = new Path(destPath+"/MLPAccuracy.txt")
    if (fs.exists(path)) {
      fs.delete(path, true)
    }
    val dataOutputStream: FSDataOutputStream = fs.create(path)
    val bw: BufferedWriter = new BufferedWriter(new OutputStreamWriter(dataOutputStream, "UTF-8"))

    // Generate Pipeline
    val pipeline = getMLPPipeLine

    // Generate Predictions
    val toxicPredictions = getPredictions(trainData, testData, getMLPPipeLine, "toxic")
    val severeToxicPredictions = getPredictions(trainData, testData, getMLPPipeLine,  "severe_toxic")
    val obscenePredictions = getPredictions(trainData, testData, getMLPPipeLine, "obscene")
    val threatPredictions = getPredictions(trainData, testData, getMLPPipeLine, "threat")
    val insultPredictions = getPredictions(trainData, testData, getMLPPipeLine, "insult")
    val identityHatePredictions = getPredictions(trainData, testData, getMLPPipeLine, "identity_hate")

    //please try the below code. We need to print the confusion matrix for this using MulticlassMetrics
    bw.write(SparkUtils.evaluateModel(toxicPredictions.rdd.map{ case Row(prediction: Double, label: Double) => (label, prediction) },"Toxic"))
    bw.write(SparkUtils.evaluateModel(severeToxicPredictions.rdd.map{ case Row(prediction: Double, label: Double) => (label, prediction) },"Severe Toxic"))
    bw.write(SparkUtils.evaluateModel(obscenePredictions.rdd.map{ case Row(prediction: Double, label: Double) => (label, prediction) },"Obscene"))
    bw.write(SparkUtils.evaluateModel(threatPredictions.rdd.map{ case Row(prediction: Double, label: Double) => (label, prediction) },"Threat"))
    bw.write(SparkUtils.evaluateModel(insultPredictions.rdd.map{ case Row(prediction: Double, label: Double) => (label, prediction) },"Insult"))
    bw.write(SparkUtils.evaluateModel(identityHatePredictions.rdd.map{ case Row(prediction: Double, label: Double) => (label, prediction) },"Identity Hate"))
    bw.close()
  }
}