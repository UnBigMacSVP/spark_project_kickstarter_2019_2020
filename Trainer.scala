package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{SparkSession}
import org.apache.spark.ml.feature.{RegexTokenizer,StopWordsRemover,IDF,CountVectorizer,StringIndexer,OneHotEncoderEstimator,VectorAssembler}
import org.apache.spark.ml.classification.{LogisticRegression}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator,ParamGridBuilder}

object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val ss = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()

    // Import des données
    val df_load = ss.read.parquet("C:\\Users\\Baptiste\\Desktop\\save\\save\\*")

    // Stage 1 : création regex_tokenizer
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("regex_tokenizer")

    // Stage 2 : Stop_words_remover
    val stop_words_remover = new StopWordsRemover()
      .setInputCol("regex_tokenizer")
      .setOutputCol("stop_words_remover")
      .setStopWords(Array[String]("i", "me", "my", "myself", "we", "our", "ours", "ourselves"
        , "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself"
        , "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was"
        , "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or"
        , "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before"
        , "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here"
        , "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not"
        , "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"))

    // Stage 3 : count_vectorizer
    val count_vectorizer = new CountVectorizer().setInputCol("stop_words_remover").setOutputCol("count_vectorizer").setMinDF(2)

    // Stage 4 : idf
    val idf = new IDF().setInputCol("count_vectorizer").setOutputCol("tfidf")

    // Stage 5 & 6 : string_indexer
    val string_indexer_country = new StringIndexer().setInputCol("country2").setOutputCol("country_cat").setHandleInvalid("keep")
    val string_indexer_currency = new StringIndexer().setInputCol("currency2").setOutputCol("currency_cat").setHandleInvalid("keep")

    // Stage 7&8 : one_hot_encoder
    val one_hot_encoder_estimator = new OneHotEncoderEstimator()
      .setInputCols(Array("country_cat", "currency_cat"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    // Stage 9 : vector_assembler
    val vector_assembler = new VectorAssembler()
      .setInputCols(Array("country_onehot", "currency_onehot", "tfidf", "days_campaign", "hours_prepa", "goal"))
      .setOutputCol("features")

    // Stage 10 : logistic_regression
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    // Création du pipeline
    val pipeline = new Pipeline()
    pipeline.setStages(Array(tokenizer, stop_words_remover, count_vectorizer, idf, string_indexer_country, string_indexer_currency, one_hot_encoder_estimator, vector_assembler, lr))

    // Création train/test sets
    val split = df_load.randomSplit(Array[Double](0.8, 0.2))
    val train = split(0)
    val test = split(1)

    // Entrainement du model
    val pipeline_model = pipeline.fit(train)
    val dfWithSimplePredictions = pipeline_model.transform(test)

    // Evaluation du modèle
    val classification_eval_1 = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setMetricName("f1")
      .setPredictionCol("predictions");

    val metric = classification_eval_1.evaluate(dfWithSimplePredictions)
    println("metric L1 sans cross-validation : ",metric);

    val classification_eval_2 = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setMetricName("f1")
      .setPredictionCol("predictions");

    // Création param_grid_builder
    val paramGrid = new ParamGridBuilder()
      .addGrid(count_vectorizer.minDF, Array(55.0D, 65.0D, 75.0D, 85.0D, 95.0D))
      .addGrid(lr.regParam, Array(0.01D, 1.0E-4D, 1.0E-6D, 1.0E-8D))
      .build()

    // création cross_validation
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(classification_eval_2)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)

    // Entrainement du modèle
    val cvModel = cv.fit(train)
    // prédiction sur les données de test
    val pred = cvModel.transform(test)

    // Evaluation du modèle
    val metric_grid = classification_eval_2.evaluate(pred)
    println("metric L1 après cross-validation : ",metric_grid)

    // Sauvegarde du modèle
    cvModel.write.overwrite().save("C:\\Users\\Baptiste\\Desktop\\save\\model\\model");
    ss.stop();

  }
}
