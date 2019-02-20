(ns clj-ml-wkg.ames-house-prices
  (:require [tech.ml.dataset.etl :as etl]
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
            [tech.io :as io]))


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
    (io/put-nippy! "file://ames-results.nippy"
                   results)
    results))


(defn results->accuracy-dataset
  [gridsearch-results]
  (->> gridsearch-results
       (map (fn [{:keys [average-loss options predict-time train-time]}]
              {:average-loss average-loss
               :model-name (str (:model-type options))
               :predict-time predict-time
               :train-time train-time}))))


(comment
  (defn accuracy-graphs
    [gridsearch-results]
    (->> [:div
          [:h1 "ames-initial"]
          [:vega-lite {:repeat {:column [:predict-time :train-time]}
                       :spec {:data {:values (results->accuracy-dataset
                                              gridsearch-results)}
                              :mark :point
                              :encoding {:y {:field :average-loss
                                             :type :quantitative}
                                         :x {:field {:repeat :column}
                                             :type :quantitative}
                                         :color {:field :model-name
                                                 :type :nominal}
                                         :shape {:field :model-name
                                                 :type :nominal}}}}]]
         oz/view!)))
