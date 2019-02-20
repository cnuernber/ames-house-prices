(ns clj-ml-wkg.ames-house-prices
  (:require [tech.io :as io]
            [camel-snake-kebab.core :refer [->kebab-case]]
            [tech.parallel :as parallel]
            [clojure.core.matrix.macros :refer [c-for]]
            [tech.ml.dataset :as dataset]
            [clojure.set :as c-set]
            [tech.xgboost]
            [tech.smile.regression]
            [tech.svm]
            [tech.ml-base :as ml]
            [tech.ml.utils :as ml-utils]
            [tech.ml.loss :as loss]
            [tech.datatype :as dtype]
            [oz.core :as oz]
            )
  (:import [tech.tablesaw.api Table ColumnType
            NumericColumn DoubleColumn
            StringColumn BooleanColumn]
           [tech.tablesaw.columns Column]
           [tech.tablesaw.io.csv CsvReadOptions]
           [java.util UUID]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)



(defn ^tech.tablesaw.io.csv.CsvReadOptions$Builder
  ->csv-builder [^String path & {:keys [separator header? date-format]}]
  (if separator
    (doto (CsvReadOptions/builder path)
      (.separator separator)
      (.header (boolean header?)))
    (doto (CsvReadOptions/builder path)
      (.header (boolean header?)))))


(defn ->table
  ^Table [path & {:keys [separator quote]}]
  (-> (Table/read)
      (.csv (->csv-builder path :separator separator :header? true))))


(def load-dataset
  (memoize
   #(->table "data/ames-house-prices/train.csv")))


(defn ->column-seq
  [item]
  (if (instance? Table item)
    (.columns ^Table item)
    (seq item)))



(defn column->seq
  [^Column col]
  (->> (.asList col)
       seq))


(defn column->unique-set
  [^Column col]
  (->> (.unique col)
       (.asList)
       set))


(defn column->metadata
  [^Column col]
  (let [num-unique (.countUnique col)]
    (merge
     {:name (.name col)
      :type (->kebab-case (.name (.type col)))
      :size (.size col)
      :num-missing (.countMissing col)
      :num-unique num-unique
      })))


(defn column-name
  [^Column item]
  (.name item))


(defn col-seq->map
  [col-seq]
  (->> (->column-seq col-seq)
       (map (juxt column-name identity))
       (into {})))


(defn update-column
  [dataset column-name f & args]
  (let [new-map (apply update (col-seq->map dataset) column-name f args)]
    (->> (->column-seq dataset)
         (map (comp #(get new-map %) #(.name ^Column %))))))


(defn column-double-op
  [dataset column-name double-fn]
  (update-column
   dataset column-name
   (fn [^Column col]
     (let [^DoubleColumn double-col (.asDoubleColumn ^NumericColumn col)]
       (parallel/parallel-for
        idx (.size double-col)
        (.set double-col idx
              (double (double-fn (.getDouble double-col idx)))))
       double-col))))


(defn log1p
  [col-name dataset]
  (column-double-op dataset col-name #(Math/log (+ 1.0 (double %)))))


(defn numeric-missing?
  [dataset]
  (->> (->column-seq dataset)
       (filter #(and (instance? NumericColumn %)
                     (> (.countMissing ^Column %) 0)))))


(defn non-numeric-missing?
  [dataset]
  (->> (->column-seq dataset)
       (filter #(and (not (instance? NumericColumn %))
                     (> (.countMissing ^Column %) 0)))))


(defn col-map
  [map-fn & args]
  (apply map map-fn (map ->column-seq args)))


(defn update-strings
  [str-fn dataset]
  (col-map (fn [^Column col]
             (if (= "string" (:type (column->metadata col)))
                (let [^"[Ljava.lang.String;" str-data (make-array String (.size col))]
                  (parallel/parallel-for
                   idx (.size col)
                   (aset str-data idx (str (str-fn (.getString ^StringColumn col idx)))))
                  (StringColumn/create (.name col) str-data))
                col))
           dataset))

(def col-type->datatype-map
  {"short" :int16
   "integer" :int32})

(defn col-datatype-cast
  [data-val ^Column column]
  (let [column-dtype-name (-> column
                              (.type)
                              (.name)
                              ->kebab-case)]
    (if-let [dtype (get col-type->datatype-map column-dtype-name)]
      (dtype/cast data-val dtype)
      (throw (ex-info "Failed to map numeric datatype to datatype library"
                      {:column-type column-dtype-name})))))

(defn update-numeric-missing
  [num-val dataset]
  (col-map (fn [^Column col]
             (if (and (instance? NumericColumn col)
                      (> (.countMissing col) 0))
               (let [new-col (.copy col)
                     ^ints missing-data (.toArray (.isMissing new-col))]
                 (parallel/parallel-for
                  idx (alength missing-data)
                  (.set new-col
                        (aget missing-data idx)
                        (col-datatype-cast (long num-val) col)))
                 new-col)
               col))
           dataset))


(defn load-ames-dataset
  []
  (->> (load-dataset)
       (update-strings #(if (= "" %)
                          "NA"
                          %))
       ;;There are only three columns that are numeric and have missing data.
       ;;Looking at the descriptions, 0 makes more sense than the column-median.
       (update-numeric-missing 0)
       (log1p "SalePrice")))


;;numeric columns that should be string columns
(def categorical-column-names
  #{"MSSubClass" "OverallQual" "OverallCond"})


(defn ->keyword-name
  [^String val]
  (keyword (->kebab-case val)))


(defn- get-column
  ^Column [column-map entry-kwd]
  (if-let [retval (get column-map entry-kwd)]
    (:column retval)
    (throw (ex-info (format "Failed to find column %s" entry-kwd)
                    {:columns (set (keys column-map))}))))


(def ames-column-names #{"PoolQC"
                          "Heating"
                          "TotalBsmtSF"
                          "HouseStyle"
                          "FullBath"
                          "YearRemodAdd"
                          "BsmtCond"
                          "Fence"
                          "Neighborhood"
                          "LotFrontage"
                          "YrSold"
                          "BldgType"
                          "PoolArea"
                          "GarageCars"
                          "BsmtFinSF2"
                          "RoofMatl"
                          "YearBuilt"
                          "GarageQual"
                          "SalePrice"
                          "LowQualFinSF"
                          "GrLivArea"
                          "Alley"
                          "LandSlope"
                          "Electrical"
                          "SaleType"
                          "PavedDrive"
                          "GarageArea"
                          "BsmtFinType2"
                          "Street"
                          "MSSubClass"
                          "WoodDeckSF"
                          "GarageFinish"
                          "ExterQual"
                          "Exterior2nd"
                          "RoofStyle"
                          "Condition1"
                          "KitchenAbvGr"
                          "BsmtFinType1"
                          "MoSold"
                          "Exterior1st"
                          "FireplaceQu"
                          "Fireplaces"
                          "LotConfig"
                          "CentralAir"
                          "GarageType"
                          "3SsnPorch"
                          "MiscFeature"
                          "Foundation"
                          "OverallCond"
                          "LotShape"
                          "BedroomAbvGr"
                          "Condition2"
                          "1stFlrSF"
                          "EnclosedPorch"
                          "MiscVal"
                          "HeatingQC"
                          "KitchenQual"
                          "2ndFlrSF"
                          "GarageCond"
                          "TotRmsAbvGrd"
                          "GarageYrBlt"
                          "BsmtHalfBath"
                          "OpenPorchSF"
                          "BsmtFinSF1"
                          "LandContour"
                          "LotArea"
                          "MasVnrArea"
                          "ScreenPorch"
                          "MasVnrType"
                          "BsmtFullBath"
                          "BsmtUnfSF"
                          "MSZoning"
                          "BsmtQual"
                          "SaleCondition"
                          "ExterCond"
                          "HalfBath"
                          "Utilities"
                          "Id"
                          "BsmtExposure"
                          "Functional"
                          "OverallQual"})


(def feature-names (c-set/difference ames-column-names #{"Id" "SalePrice"}))

(def label-name "SalePrice")


(defn- checknan
  ^double [col-kwd row-idx item]
  (let [retval (double item)]
    (when (Double/isNaN retval)
      (throw (ex-info (format "NAN detected in column: %s[%s]" col-kwd row-idx) {})))
    retval))


(defn set-string-table
  ^StringColumn [^StringColumn column str-table]
  (when-not (= (column->unique-set column)
               (set (keys str-table)))
    (throw (ex-info "String table keys existing unique set mismatch"
                    {:str-table-keys (set (keys str-table))
                     :column-unique-set (column->unique-set column)})))
  (let [new-str-table (StringColumn/create (.name column))]
    ;;uggh.  Create the appropriate type for the size of the unique-set
    ;;and build the forward/reverse mappings.
    ;;Then set all the values and you should have it.
    )

  )



;; Tech datasets are essentially row store.
;; Coalesced datasets are dense vectors of data
;; There is a tension here between the column store and row store format
;; that is deeper than I was thinking.  For training, row store makes sense
;; but in my experience data coming in from an API is a sequence of maps and ths
;; you really want some abstraction where you can define some pipeline that applies
;; to either.
(defn ->tech-ml-dataset
  [{:keys [label-map] :as options} dataset]
  (let [column-map (->> (->column-seq dataset)
                        (map (fn [^Column col]
                               (let [col-name (->keyword-name (.name col))]
                                 [col-name (-> (column->metadata col)
                                               (assoc :name col-name
                                                      :column col))])))
                        (into {}))
        label-map (merge (->> (vals column-map)
                              (filter #(= "string" (get % :type)))
                              (map (fn [{:keys [column name]}]
                                     (let [^StringColumn column column]
                                       [name (->> (column->unique-set column)
                                                  (map-indexed #(vector %2 %1))
                                                  (into {}))])))
                              (into {}))
                         ;;Allow users to override string->int mapping
                         label-map)
        column-sizes (->> (->column-seq dataset)
                          (map #(.size ^Column %))
                          distinct)

        _  (when (> 1 (count column-sizes))
             (throw (ex-info "Mismatched column sizes" {:column-sizes column-sizes})))

        feature-names (->> (mapv ->keyword-name feature-names))
        label-name (->keyword-name label-name)
        n-features (count feature-names)
        col-getter-fn (fn [col-kwd]
                        (let [column (get-column column-map col-kwd)
                              label-entry (get label-map col-kwd)]
                          (if label-entry
                            (fn [row-idx]
                              (let [retval (get label-entry (.get ^Column column (int row-idx)))]
                                (checknan col-kwd row-idx retval)))
                            (cond
                              (instance? NumericColumn column)
                              (fn [row-idx]
                                (checknan
                                 col-kwd row-idx
                                 (.getDouble ^NumericColumn column (int row-idx))))
                              (instance? BooleanColumn column)
                              (fn [row-idx]
                                (checknan
                                 col-kwd row-idx
                                 (.getDouble ^BooleanColumn column (int row-idx))))))))
        feature-columns (mapv col-getter-fn feature-names)
        label-column (col-getter-fn label-name)
        options (assoc options :label-map label-map)
        key-ecount-map (->> feature-names
                            (map #(vector % 1))
                            (into {}))]

    (->> (range (first column-sizes))
         (map (fn [row-idx]
                (let [feature-data (double-array n-features)]
                  (c-for [idx 0 (< idx n-features) (inc idx)]
                         (aset feature-data idx (double ((get feature-columns idx) row-idx))))
                  {::dataset/features feature-data
                   ::dataset/label (double-array [(label-column row-idx)])})))
         (dataset/post-process-coalesced-dataset options
                                                 feature-names
                                                 key-ecount-map
                                                 [label-name]))))


(defn gridsearch-model
  [dataset-name dataset loss-fn opts]
  (let [gs-options (ml/auto-gridsearch-options opts)]
    (println (format "Dataset: %s, Model %s"
                     dataset-name
                     (:model-type opts)))
    (let [gs-start (System/nanoTime)
          {results :retval
           milliseconds :milliseconds}
          (ml-utils/time-section
           (apply ml/gridsearch
                  [gs-options]
                  [::dataset/features]
                  [::dataset/label]
                  loss-fn dataset
                  :gridsearch-depth 75
                  :top-n 20
                  (apply concat (seq opts))))]
      (->> results
           (mapv #(merge %
                         {:gridsearch-time-ms milliseconds
                          :gridsearch-id (UUID/randomUUID)
                          :dataset-name dataset-name}))))))



(defn do-gridsearch
  [base-systems result-name]
  (let [{options :options
         dataset :coalesced-dataset} (->> (load-ames-dataset)
                                          (->tech-ml-dataset {:range-map {::dataset/features [-1 1]}}))
        keyset (set (keys (first dataset)))
        feature-keys (disj keyset :Purchase)
        results (->>  base-systems
                      (map #(merge options %))
                      (mapcat
                       (partial gridsearch-model
                                result-name
                                dataset
                                loss/rmse))
                      vec)]
    (io/put-nippy! (format
                    "file://%s-results.nippy"
                    (name result-name))
                   results)
    results))


(defn gridsearch-the-things
  []
  (do-gridsearch [{:model-type :xgboost/regression}
                  {:model-type :smile.regression/lasso}
                  {:model-type :smile.regression/ridge}
                  {:model-type :smile.regression/elastic-net}
                  {:model-type :libsvm/regression}]
                 :ames-initial))

(defn results->accuracy-dataset
  [gridsearch-results]
  (->> gridsearch-results
       (map (fn [{:keys [average-loss options predict-time train-time]}]
              {:average-loss average-loss
               :model-name (str (:model-type options))
               :predict-time predict-time
               :train-time train-time}))))

(defn accuracy-graphs
  [gridsearch-results]
  (->> [:div
        [:h1 "ames-initial"]
        [:vega-lite {:repeat {:column [:predict-time :train-time]}
                     :spec {:data {:values (results->accuracy-dataset gridsearch-results)}
                            :mark :point
                            :encoding {:y {:field :average-loss
                                           :type :quantitative}
                                       :x {:field {:repeat :column}
                                           :type :quantitative}
                                       :color {:field :model-name
                                               :type :nominal}
                                       :shape {:field :model-name
                                               :type :nominal}}}}]]
       oz/view!))
