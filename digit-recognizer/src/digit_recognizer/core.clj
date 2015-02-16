(ns digit-recognizer.core
  (:require [k-nn.core :as knn]
            [clojure.string :as str])
  (:gen-class))

;num training cases total in train.csv 42000
;40k training 2k testing

(defn parse-data [data]
  ;sum every 28, should result in 28 features
  (flatten (map (partial reduce +) (partition-all 5 (map #(Integer/parseInt %) data)))))

(defn build-neighborhood [rdr]
  (map (fn [line]
         (let [[label & data] (str/split line #",")]
           {:class label
            :features (parse-data data)}))
       (take 10000 (rest (line-seq rdr)))))

(defn classify [rdr neighborhood]
  (doall
    (map (fn [line]
           (let [[answer & data] (str/split line #",")]
             [(:class (knn/classify 1 neighborhood (parse-data data)))
              answer]))
         (take 1000 (line-seq rdr)))))

(defn get-accuracy [results]
  (/ (reduce +
             (map (fn [[prediction actual]]
                    (if (= prediction actual) 1 0))
                  results))
     (count results)))

(defn -main
  "Parse training data from train.csv to build k-nn model for test.csv"
  [& args]
  (time
    (with-open [rdr (clojure.java.io/reader "train.csv")]
      (let [neighborhood (build-neighborhood rdr)
            results (classify rdr neighborhood)]
        (println (get-accuracy results))))))
