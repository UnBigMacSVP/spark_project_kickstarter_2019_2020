package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{udf,concat,lit,lower,when,round}

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Configuration spark
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession
    val ss = SparkSession.builder.config(conf).appName("TP Spark : Preprocessor").getOrCreate()

    // Import implicits
    import ss.implicits._

    val df_imported =  ss
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("C:\\Users\\Baptiste\\Desktop\\train_clean.csv")


    // Définition de la fonction UDF de nettoyage de la colonne Country
    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else
        country
    }
    val cleanCountryUdf = udf(cleanCountry _)

    // Définition de la fonction UDF de nettoyage de la colonne Country
    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }
    val cleanCurrencyUdf = udf(cleanCurrency _)

    // Nettoyage des données
    val df_final = df_imported
      .withColumn("goal", $"goal".cast("Int")) // cast de la colonne $goal en Int
      .withColumn("final_status", $"final_status".cast("Int")) // cast de la colonne $final_status en Int
      .where($"final_status" === 0 || $"final_status" === 1) // suppression des final_status autre que 0 ou 1
      .withColumn("country2", cleanCountryUdf($"country", $"currency")) // Nettoyage de la colonne $country
      .withColumn("currency2", cleanCurrencyUdf($"currency")) // Nettoyage de la colonne $currency
      .withColumn("days_campaign",round(($"deadline" - $"launched_at")/86400)) // création de la colonne "days_campaign"
      .withColumn("hours_prepa",round(($"launched_at" - $"created_at")/3600)) // création de la colonne "hours_prepa"
      .where($"country2" =!= "True") // Suppression valeurs non pertinentes
      .where($"country2" =!= "DE") // Suppression valeurs non pertinentes
      .where($"currency2" =!= "null") // Suppression valeurs non pertinentes
      .where($"goal" > 10 ) // Suppression valeurs non pertinentes
      .withColumn("country2", when($"country2" === null, "unknown").otherwise($"country2")) //suppression des valeurs nulles
      .withColumn("currency2", when($"currency2" === null, "unknown").otherwise($"currency2")) //suppression des valeurs nulles
      .withColumn("goal", when($"goal" === null, -1).otherwise($"goal")) //suppression des valeurs nulles
      .withColumn("hours_prepa", when($"hours_prepa" === null, -1).otherwise($"hours_prepa")) //suppression des valeurs nulles
      .withColumn("days_campaign", when($"days_campaign" === null, -1).otherwise($"days_campaign")) //suppression des valeurs nulles
      .withColumn("text", concat(lower($"desc"),lit(" "), lower($"name"),lit(" "), lower($"keywords"))) // création de la colonne "text"
      .where($"text" !== "null")
      .drop("backers_count","desc","project_id","name","keywords", "state_changed_at","country","currency","disable_communication","deadline","launched_at","created_at")

    println(s"Nombre de lignes : ${df_final.count}")
    println(s"Nombre de colonnes : ${df_final.columns.length}")

    df_final.printSchema()
    df_final.show()

    // Sauvegarde du modèle
    df_final.write.parquet("C:\\Users\\Baptiste\\Desktop\\save\\save")
  }
}
