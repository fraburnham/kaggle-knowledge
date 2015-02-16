(ns digit-recognizer.core
  (:require [k-nn.core :as knn]
            [clojure.string :as str])
  (:gen-class))

;num training cases total in train.csv 42000
;40k training 2k testing


(defn build-neighborhood [rdr]
  (map (fn [line]
         (let [[label & data] (str/split line #",")]
           {:class label
            :features (map #(Integer/parseInt %) data)}))
       (take 40000 (rest (line-seq rdr)))))

(defn classify [rdr neighborhood]
  (doall
    (map (fn [line]
           (let [[answer & data] (str/split line #",")]
             (println "Prediction: " (knn/classify 1 neighborhood
                                                   (map #(Integer/parseInt %) data))
                      "Actual: " answer))) (line-seq rdr))))

(defn -main
  "Parse training data from train.csv to build k-nn model for test.csv"
  [& args]
  (time
    (with-open [rdr (clojure.java.io/reader "train.csv")]
      (let [neighborhood (build-neighborhood rdr)]
        (classify rdr neighborhood)))))
