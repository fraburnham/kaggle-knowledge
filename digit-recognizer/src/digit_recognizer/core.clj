(ns digit-recognizer.core
  (:require [k-nn.core :as knn]
            [clojure.string :as str])
  (:gen-class))

;num training cases total in train.csv 42000
;40k training 2k testing

(defmacro n? [r]
  `(if (contains? ~r :n) (inc (:n ~r)) 2))

(defn parse-data [data]
  (map #(Integer/parseInt %) data))

(defn cma [n last-cma x]
  (/ (+ x (* n last-cma)) (inc n)))

(defn avg-data [neighborhood]
  (map
    #(reduce
      (fn [r x]
        {:class (:class x)
         :features (map (partial cma (n? r)) (:features r) (:features x))
         :n (n? r)})
      %)
    (map #(get neighborhood %)
         (keys neighborhood))))

(defn build-neighborhood [rdr]
  ;now group by class and average
  (group-by :class
            (map (fn [line]
                   (let [[label & data] (str/split line #",")]
                     {:class label
                      :features (parse-data data)}))
                 (take 15000 (rest (line-seq rdr))))))

(defn classify [rdr neighborhood]
  (doall
    (map (fn [line]
           (let [[answer & data] (str/split line #",")]
             [(:class (knn/classify 1 neighborhood (parse-data data)))
              answer]))
         (take 100 (line-seq rdr)))))

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
      (let [neighborhood (avg-data (build-neighborhood rdr))
            results (classify rdr neighborhood)]
        (println (get-accuracy results))))))
