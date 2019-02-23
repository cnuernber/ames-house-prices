(ns clj-ml-wkg.ames-house-prices
  "See reference article:
  https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset

  This is very much a work in progress."
  (:require [tech.ml.dataset.etl :as etl]
            [tech.ml.dataset.etl.pipeline-operators :as pipe-ops]
            [tech.ml.dataset.etl.math-ops :as pipe-math]
            [tech.ml.dataset.etl.column-filters :as col-filters]
            [tech.ml.dataset :as dataset]
            [tech.ml.dataset.column :as ds-col]
            [tech.ml :as ml]
            [tech.ml.loss :as loss]
            [tech.ml.utils :as ml-utils]

            ;;use tablesaw as dataset backing store
            [tech.libs.tablesaw :as tablesaw]

            ;;model generators
            [tech.libs.xgboost]
            [tech.libs.smile.regression]
            [tech.libs.svm]

            ;;put/get nippy
            [tech.io :as io]

            [oz.core :as oz]

            [clojure.pprint :as pp])

  (:import [tech.ml.protocols.etl PETLSingleColumnOperator]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)



(def full-ames-pt-1
  '[[remove "Id"]
    ;;Replace missing values or just empty csv values with NA
    [replace-missing string? "NA"]
    [replace-string string? "" "NA"]
    [replace-missing numeric? 0]
    [replace-missing boolean? false]
    [->etl-datatype [or numeric? boolean?]]
    [string->number "Utilities" [["NA" -1] "ELO" "NoSeWa" "NoSewr" "AllPub"]]
    [string->number "LandSlope" ["Gtl" "Mod" "Sev" "NA"]]
    [string->number ["ExterQual"
                     "ExterCond"
                     "BsmtQual"
                     "BsmtCond"
                     "HeatingQC"
                     "KitchenQual"
                     "FireplaceQu"
                     "GarageQual"
                     "GarageCond"
                     "PoolQC"]   ["NA" "Po" "Fa" "TA" "Gd" "Ex"]]
    [set-attribute ["MSSubClass" "OverallQual" "OverallCond"] :categorical? true]
    [string->number "MasVnrType" {"BrkCmn" 1
                                 "BrkFace" 1
                                 "CBlock" 1
                                 "Stone" 1
                                 "None" 0
                                 "NA" -1}]
    [string->number "SaleCondition" {"Abnorml" 0
                                     "Alloca" 0
                                     "AdjLand" 0
                                     "Family" 0
                                     "Normal" 0
                                     "Partial" 1
                                     "NA" -1}]
    ;;Auto convert the rest that are still string columns
    [string->number string?]
    [m= "SalePrice" (log1p (col "SalePrice"))]
    [m= "OverallGrade" (* (col "OverallQual") (col "OverallCond"))]
    ;; Overall quality of the garage
    [m= "GarageGrade"  (* (col "GarageQual") (col "GarageCond"))]
    ;; Overall quality of the exterior
    [m= "ExterGrade" (* (col "ExterQual") (col "ExterCond"))]
    ;; Overall kitchen score
    [m= "KitchenScore" (* (col "KitchenAbvGr") (col "KitchenQual"))]
    ;; Overall fireplace score
    [m= "FireplaceScore" (* (col "Fireplaces") (col "FireplaceQu"))]
    ;; Overall garage score
    [m= "GarageScore" (* (col "GarageArea") (col "GarageQual"))]
    ;; Overall pool score
    [m= "PoolScore" (* (col "PoolArea") (col "PoolQC"))]
    ;; Simplified overall quality of the house
    [m= "SimplOverallGrade" (* (col "OverallQual") (col "OverallCond"))]
    ;; Simplified overall quality of the exterior
    [m= "SimplExterGrade" (* (col "ExterQual") (col "ExterCond"))]
    ;; Simplified overall pool score
    [m= "SimplPoolScore" (* (col "PoolArea") (col "PoolQC"))]
    ;; Simplified overall garage score
    [m= "SimplGarageScore" (* (col "GarageArea") (col "GarageQual"))]
    ;; Simplified overall fireplace score
    [m= "SimplFireplaceScore" (* (col "Fireplaces") (col "FireplaceQu"))]
    ;; Simplified overall kitchen score
    [m= "SimplKitchenScore" (* (col "KitchenAbvGr") (col "KitchenQual"))]
    ;; Total number of bathrooms
    [m= "TotalBath" (+ (col "BsmtFullBath")
                       (* 0.5 (col "BsmtHalfBath"))
                       (col "FullBath")
                       (* 0.5 "HalfBath"))]
    ;; Total SF for house (incl. basement)
    [m= "AllSF" (+ (col "GrLivArea") (col "TotalBsmtSF"))]
    ;; Total SF for 1st + 2nd floors
    [m= "AllFlrsSF" (+ (col "1stFlrSF") (col "2ndFlrSF"))]
    ;; Total SF for porch
    [m= "AllPorchSF" (+ (col "OpenPorchSF") (col "EnclosedPorch")
                        (col "3SsnPorch") (col "ScreenPorch"))]])

;; Found after first step using dataset/correlation-table
(def ames-top-columns ["SalePrice"
                        "OverallQual"
                        "AllSF"
                        "AllFlrsSF"
                        "GrLivArea"
                        "GarageCars"
                        "ExterQual"
                        "KitchenQual"
                        "GarageScore"
                        "SimplGarageScore"
                        "GarageArea"])


(def full-ames-pt-2
  ;;Drop SalePrice column of course.
  (->> (rest ames-top-columns)
       (mapcat (fn [colname]
                 [['m= (str colname "-s2") ['** ['col colname] 2]]
                  ['m= (str colname "-s3") ['** ['col colname] 3]]
                  ['m= (str colname "-sqrt") ['sqrt ['col colname]]]]))
       (concat full-ames-pt-1)
       vec))


(def full-ames-pt-3
  (->> (concat full-ames-pt-2
               '[[m= [and
                      [not categorical?]
                      [not target?]
                      [> [abs [skew [col]]] 0.5]]
                  (log1p (col))]

                 [std-scaler [and
                              [not categorical?]
                              [not target?]]]])
       (vec)))


(defn gridsearch-model
  [dataset-name dataset loss-fn opts]
  (let [gs-options (ml/auto-gridsearch-options
                    (assoc opts
                           :gridsearch-depth 75
                           :top-n 20))]
    (println (format "Dataset: %s, Model %s"
                     dataset-name
                     (:model-type opts)))
    (let [gs-start (System/nanoTime)
          {results :retval
           milliseconds :milliseconds}
          (ml-utils/time-section
           (ml/gridsearch
            gs-options
            loss-fn
            dataset))]
      (->> results
           (mapv #(merge %
                         {:gridsearch-time-ms milliseconds
                          :dataset-name dataset-name}))))))



(defn gridsearch-dataset
  []
  (let [base-systems [{:model-type :xgboost/regression}
                      {:model-type :smile.regression/lasso}
                      {:model-type :smile.regression/ridge}
                      {:model-type :smile.regression/elastic-net}
                      {:model-type :libsvm/regression}]

        dataset-name :full-ames-pathway

        src-dataset (tablesaw/path->tablesaw-dataset
                     "data/ames-house-prices/train.csv")

        {:keys [dataset pipeline options]}
        (etl/apply-pipeline src-dataset full-ames-pt-3
                            {:target "SalePrice"})

        results (->> base-systems
                     (map #(merge options %))
                     (mapcat
                      (partial gridsearch-model
                               dataset-name
                               dataset
                               loss/rmse))
                     vec)]
    (io/put-nippy! "file://ames-gridsearch-results.nippy"
                   results)
    results))


(def load-results
  (memoize
   (fn
     []
     (io/get-nippy "file://ames-gridsearch-results.nippy"))))


(defn results->accuracy-dataset
  [gridsearch-results]
  (->> gridsearch-results
       (map (fn [{:keys [average-loss options predict-time train-time]}]
              {:average-loss average-loss
               :model-name (str (:model-type options))
               :predict-time predict-time
               :train-time train-time}))))


(def initial-pipeline-from-article
  '[[remove "Id"]
    [m= "SalePrice" (log1p (col))]])


;; Impressive patience!!
(def initial-missing-entries
  '[
    ;; Handle missing values for features where median/mean or most common value doesn't
    ;; make sense

    ;; Alley : data description says NA means "no alley access"
    [replace-missing "Alley" "None"]
    ;; BedroomAbvGr : NA most likely means 0
    [replace-missing ["BedroomAbvGr"
                      "BsmtFullBath"
                      "BsmtHalfBath"
                      "BsmtUnfSF"
                      "EnclosedPorch"
                      "Fireplaces"
                      "GarageArea"
                      "GarageCars"
                      "HalfBath"
                      ;; KitchenAbvGr : NA most likely means 0
                      "KitchenAbvGr"
                      "LotFrontage"
                      "MasVnrArea"
                      "MiscVal"
                      ;; OpenPorchSF : NA most likely means no open porch
                      "OpenPorchSF"
                      "PoolArea"
                      ;; ScreenPorch : NA most likely means no screen porch
                      "ScreenPorch"
                      ;; TotRmsAbvGrd : NA most likely means 0
                      "TotRmsAbvGrd"
                      ;; WoodDeckSF : NA most likely means no wood deck
                      "WoodDeckSF"
                      ]
     0]
    ;; BsmtQual etc : data description says NA for basement features is "no basement"
    [replace-missing ["BsmtQual"
                      "BsmtCond"
                      "BsmtExposure"
                      "BsmtFinType1"
                      "BsmtFinType2"
                      ;; Fence : data description says NA means "no fence"
                      "Fence"
                      ;; FireplaceQu : data description says NA means "no fireplace"
                      "FireplaceQu"
                      ;; GarageType etc : data description says NA for garage features
                      ;; is "no garage"
                      "GarageType"
                      "GarageFinish"
                      "GarageQual"
                      "GarageCond"
                      ;; MiscFeature : data description says NA means "no misc feature"
                      "MiscFeature"
                      ;; PoolQC : data description says NA means "no pool"
                      "PoolQC"
                      ]
     "No"]
    [replace-missing "CentralAir" "N"]
    [replace-missing ["Condition1"
                      "Condition2"]
     "Norm"]
    ;; Condition : NA most likely means Normal
    ;; EnclosedPorch : NA most likely means no enclosed porch
    ;; External stuff : NA most likely means average
    [replace-missing ["ExterCond"
                      "ExterQual"
                      ;; HeatingQC : NA most likely means typical
                      "HeatingQC"
                      ;; KitchenQual : NA most likely means typical
                      "KitchenQual"
                      ]
     "TA"]
    ;; Functional : data description says NA means typical
    [replace-missing "Functional" "Typ"]
    ;; LotShape : NA most likely means regular
    [replace-missing "LotShape" "Reg"]
    ;; MasVnrType : NA most likely means no veneer
    [replace-missing "MasVnrType" "None"]
    ;; PavedDrive : NA most likely means not paved
    [replace-missing "PavedDrive" "N"]
    [replace-missing "SaleCondition" "Normal"]
    [replace-missing "Utilities" "AllPub"]])


(def more-categorical
  '[[set-attribute ["MSSubClass" "OverallQual" "OverallCond"] :categorical? true]])


;; Encode some categorical features as ordered numbers when there is information in the
;; order.
(def str->number-initial-map
  {
   "Alley"  {"Grvl"  1 "Pave" 2 "None" 0}
   "BsmtCond"  {"No"  0 "Po"  1 "Fa"  2 "TA"  3 "Gd"  4 "Ex"  5}
   "BsmtExposure"  {"No"  0 "Mn"  1 "Av" 2 "Gd"  3}
   "BsmtFinType1"  {"No"  0 "Unf"  1 "LwQ" 2 "Rec"  3 "BLQ"  4
                     "ALQ"  5 "GLQ"  6}
   "BsmtFinType2"  {"No"  0 "Unf"  1 "LwQ" 2 "Rec"  3 "BLQ"  4
                     "ALQ"  5 "GLQ"  6}
   "BsmtQual"  {"No"  0 "Po"  1 "Fa"  2 "TA" 3 "Gd"  4 "Ex"  5}
   "ExterCond"  {"Po"  1 "Fa"  2 "TA" 3 "Gd" 4 "Ex"  5}
   "ExterQual"  {"Po"  1 "Fa"  2 "TA" 3 "Gd" 4 "Ex"  5}
   "FireplaceQu"  {"No"  0 "Po"  1 "Fa"  2 "TA"  3 "Gd"  4 "Ex"  5}
   "Functional"  {"Sal"  1 "Sev"  2 "Maj2"  3 "Maj1"  4 "Mod" 5
                   "Min2"  6 "Min1"  7 "Typ"  8}
   "GarageCond"  {"No"  0 "Po"  1 "Fa"  2 "TA"  3 "Gd"  4 "Ex"  5}
   "GarageQual"  {"No"  0 "Po"  1 "Fa"  2 "TA"  3 "Gd"  4 "Ex"  5}
   "HeatingQC"  {"Po"  1 "Fa"  2 "TA"  3 "Gd"  4 "Ex"  5}
   "KitchenQual"  {"Po"  1 "Fa"  2 "TA"  3 "Gd"  4 "Ex"  5}
   "LandSlope"  {"Sev"  1 "Mod"  2 "Gtl"  3}
   "LotShape"  {"IR3"  1 "IR2"  2 "IR1"  3 "Reg"  4}
   "PavedDrive"  {"N"  0 "P"  1 "Y"  2}
   "PoolQC"  {"No"  0 "Fa"  1 "TA"  2 "Gd"  3 "Ex"  4}
   "Street"  {"Grvl"  1 "Pave"  2}
   "Utilities"  {"ELO"  1 "NoSeWa"  2 "NoSewr"  3 "AllPub"  4}
   })


(def str->number-pipeline
  (->> str->number-initial-map
       (map (fn [[k v-map]]
              ['string->number k v-map]))))


(def replace-maps
  {
   ;; Create new features
   ;; 1* Simplifications of existing features
   "SimplOverallQual" {"OverallQual" {1  1, 2  1, 3  1, ;; bad
                                     4  2, 5  2, 6  2, ;; average
                                     7  3, 8  3, 9  3, 10  3 ;; good
                                      }}
   "SimplOverallCond" {"OverallCond" {1  1, 2  1, 3  1,       ;; bad
                                      4  2, 5  2, 6  2,       ;; average
                                      7  3, 8  3, 9  3, 10  3 ;; good
                                      }}
   "SimplPoolQC" {"PoolQC" {1  1, 2  1,    ;; average
                            3  2, 4  2     ;; good
                            }}
   "SimplGarageCond" {"GarageCond" {1  1,             ;; bad
                                    2  1, 3  1,       ;; average
                                    4  2, 5  2        ;; good
                                    }}
   "SimplGarageQual" {"GarageQual" {1  1,             ;; bad
                                    2  1, 3  1,       ;; average
                                    4  2, 5  2        ;; good
                                    }}
   "SimplFireplaceQu"  {"FireplaceQu" {1  1,           ;; bad
                                        2  1, 3  1,     ;; average
                                        4  2, 5  2      ;; good
                                        }}
   "SimplFunctional"  {"Functional" {1  1, 2  1,           ;; bad
                                      3  2, 4  2,           ;; major
                                      5  3, 6  3, 7  3,     ;; minor
                                      8  4                  ;; typical
                                      }}
   "SimplKitchenQual" {"KitchenQual" {1  1,             ;; bad
                                      2  1, 3  1,       ;; average
                                      4  2, 5  2        ;; good
                                      }}
   "SimplHeatingQC"  {"HeatingQC" {1  1,           ;; bad
                                   2  1, 3  1,     ;; average
                                   4  2, 5  2      ;; good
                                    }}
   "SimplBsmtFinType1"  {"BsmtFinType1" {1  1,         ;; unfinished
                                         2  1, 3  1,   ;; rec room
                                         4  2, 5  2, 6  2 ;; living quarters
                                          }}
   "SimplBsmtFinType2" {"BsmtFinType2" {1 1,           ;; unfinished
                                        2 1, 3 1,      ;; rec room
                                        4 2, 5 2, 6 2  ;; living quarters
                                        }}
   "SimplBsmtCond" {"BsmtCond" {1 1,    ;; bad
                                2 1, 3 1, ;; average
                                4 2, 5 2  ;; good
                                }}
   "SimplBsmtQual" {"BsmtQual" {1 1,      ;; bad
                                2 1, 3 1, ;; average
                                4 2, 5 2  ;; good
                                }}
   "SimplExterCond" {"ExterCond" {1 1,      ;; bad
                                  2 1, 3 1, ;; average
                                  4 2, 5 2  ;; good
                                  }}
   "SimplExterQual" {"ExterQual" {1 1,      ;; bad
                                  2 1, 3 1, ;; average
                                  4 2, 5 2  ;; good
                                  }}
   })


(def simplifications
  (->> replace-maps
       (mapv (fn [[k v-map]]
               (let [[src-name replace-data] (first v-map)]
                 ['m= k ['replace ['col src-name] replace-data]])))))


(defn pp-str
  [ds]
  (with-out-str
    (pp/pprint ds)))


(defn presentation
  []
  (let [src-dataset (tablesaw/path->tablesaw-dataset
                     "data/ames-house-prices/train.csv")
        outliers-graph
        [:vega-lite {:data {:values
                            (-> src-dataset
                                (dataset/select ["SalePrice" "GrLivArea"] :all)
                                (dataset/->flyweight))}
                     :mark :point
                     :encoding {:y {:field "SalePrice"
                                    :type :quantitative}
                                :x {:field "GrLivArea"
                                    :type :quantitative}}}]
        filtered-ds (pipe-ops/filter src-dataset "GrLivArea" '(< (col) 4000))
        fixed-outlier-graph
        [:vega-lite {:data {:values
                            (-> filtered-ds
                                (dataset/select ["SalePrice" "GrLivArea"] :all)
                                (dataset/->flyweight))}
                     :mark :point
                     :encoding {:y {:field "SalePrice"
                                    :type :quantitative}
                                :x {:field "GrLivArea"
                                    :type :quantitative}}}]
        after-categorical (-> (etl/apply-pipeline filtered-ds
                                                  (concat initial-pipeline-from-article
                                                          more-categorical)
                                                  {})
                              :dataset)
        missing-ds-1 (-> (etl/apply-pipeline filtered-ds
                                             (concat initial-pipeline-from-article
                                                     more-categorical
                                                     initial-missing-entries)
                                             {})
                         :dataset)

        str->num-1-data (etl/apply-pipeline filtered-ds
                                            (concat initial-pipeline-from-article
                                                    more-categorical
                                                    initial-missing-entries
                                                    str->number-pipeline)
                                            {})

        simplified-data (-> (etl/apply-pipeline filtered-ds
                                                (concat initial-pipeline-from-article
                                                        more-categorical
                                                        initial-missing-entries
                                                        str->number-pipeline
                                                        simplifications)
                                                {})
                            :dataset)]
    (->> [:div
          [:h1 "Ames House Prices"]
          [:div
           [:h3 "Outliers"]
           [:p "First we note that there are several dataset outliers"]
           outliers-graph
           [:p "We then fix this with a simple pipeline operation: "
            [:pre (pr-str initial-pipeline-from-article)]]
           fixed-outlier-graph]
          [:div
           [:h3 "Categorical Fixes"]
           [:p "Whether columns are categorical is defined by attributes:"]
           [:p "Pre-fix:"
            [:pre (pp-str (col-filters/execute-column-filter
                           src-dataset 'categorical?))]]
           [:p "The fix is simple:"
            [:pre (pp-str more-categorical)]]
           [:p "Post-fix:"
            [:pre (pp-str (col-filters/execute-column-filter
                           after-categorical 'categorical?))]]]
          [:div
           [:h3 "Missing #1"]
           [:p "Initial missing columns:"
            [:pre
             (pp-str
               (dataset/columns-with-missing-seq filtered-ds))]]
           [:p "Adding in pipeline ops:"
            [:pre (pp-str initial-missing-entries)]]
           [:p "After initial missing fixes"
            [:pre
             (pp-str (dataset/columns-with-missing-seq missing-ds-1))]]]
          [:div
           [:h3 "string->number"]
           [:p "String->number is flexible and remembers what it did for reverse
mapping and inference.  Pipeline:"
            [:pre (pp-str str->number-pipeline)]]
           [:p "The options map returned contains the 'label-map'.  This is used
throughout the system in order to map values both ways so we can always recover
the original string value."
            [:pre (pp-str (select-keys (:options str->num-1-data) [:label-map]))]]]

          [:div
           [:h3 "simplified data"]
           [:p "The math system has a special function replace which takes a column
and a map and returns a new column."
            [:pre (pp-str '(-> (dataset/column simplified-data "KitchenQual")
                               (ds-col/unique)))]
            [:pre (pp-str (-> (dataset/column simplified-data "KitchenQual")
                              (ds-col/unique)))]]
           [:p "We run a math expression and place result into new column"
            [:pre (pp-str (first simplifications))]]
           [:p "This results (in this case) in a column with fewer values."
            [:pre (pp-str '(-> (dataset/column simplified-data "SimplKitchenQual")
                               (ds-col/unique)))]
            [:pre (pp-str (-> (dataset/column simplified-data "SimplKitchenQual")
                              (ds-col/unique)))]]]

          [:h3 "Overall Gridsearch Results"]
          [:vega-lite {:data {:values (results->accuracy-dataset
                                       (load-results))}
                       :mark :point
                       :encoding {:y {:field :average-loss
                                      :type :quantitative}
                                  :x {:field :model-name
                                      :type :nominal}
                                  :color {:field :model-name
                                          :type :nominal}
                                  :shape {:field :model-name
                                          :type :nominal}}}]]
         oz/view!)))
