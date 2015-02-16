(ns digit-recognizer.core
  (:require [k-nn.core :as knn]
            [clojure.string :as str])
  (:gen-class))

(defn parse-data [data]
  (flatten (map (partial reduce +)
                (partition-all 5 (map #(Integer/parseInt %) data)))))

(defn build-neighborhood [rdr]
  (map (fn [line]
         (let [[label & data] (str/split line #",")]
           {:class label
            :features (parse-data data)}))
       (take 10000 (rest (line-seq rdr)))))

(defn classify [rdr neighborhood]
  (map (fn [line]
         (let [data (str/split line #",")]
           (:class (knn/classify 1 neighborhood (parse-data data)))))
       (rest (line-seq rdr))))

(defn file-dump [results]
  (with-open [dump (clojure.java.io/writer "submission.csv")]
    (.write dump (str "ID,label\n"))
    (loop [res results
           id 1]
      (if (not (empty? res))
        (do
          (.write dump (str id "," (first (first res)) "\n"))
          (recur (rest res) (inc id)))))))

(defn -main
  "Parse training data from train.csv to build k-nn model for test.csv"
  [& args]
  (time
    (with-open [train (clojure.java.io/reader "train.csv")
                test (clojure.java.io/reader "test.csv")]
      (let [neighborhood (build-neighborhood train)
            results (classify test neighborhood)]
        (file-dump results)))))
