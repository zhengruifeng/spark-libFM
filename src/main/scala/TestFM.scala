
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.{Logging, SparkConf, SparkContext}
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils


object TestFM extends App with Logging {

  override def main(args: Array[String]): Unit = {

    val sparkConf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("TestFM")
    val sc = new SparkContext(sparkConf)

    val trainDataset = MLUtils.loadLibSVMFile(sc, "data/ml-100k.train.libfm").cache()
    val testDataset = MLUtils.loadLibSVMFile(sc, "data/ml-100k.test.libfm").cache()

    //    val task = args(1).toInt
    //    val numIterations = args(2).toInt
    //    val stepSize = args(3).toDouble
    //    val miniBatchFraction = args(4).toDouble

    val fm = FMWithSGD.train(
      trainDataset, task = 1,
      numIterations = 10,
      stepSize = 0.1,
      miniBatchFraction = 1.0,
      dim = (true, true, 4),
      regParam = (0, 0, 0),
      initStd = 0.01)

    val predictionsAndLabels = testDataset.map(x => (fm.predict(x.features), x.label))
    val metrics = new RegressionMetrics(predictionsAndLabels)
    logInfo(s"metrics.rootMeanSquaredError: ${metrics.rootMeanSquaredError}")
    logInfo(s"metrics.r2: ${metrics.r2}")
  }
}
