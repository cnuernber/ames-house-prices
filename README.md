# aimes-house-prices

Exploration of kaggle aimes house prices comp.

## Usage

From top level directory:
```
scripts/get-data.sh
```

The data is under data/aimes-house-prices


Make sure you have openblas/atlas isntalled as well as libsvm.

If SVM doesn't work just remove it from the gridsearch pathways.  It is an old C
library that can be temperamental.  From ubuntu it works fine on jvm 8.


There is some dependency conflict with the csv subsystem tablesaw so from here is the workflow:

```clojure
lein repl

;;load the aimes namespace
(require '[clj-ml-wkg/aimes-house-prices])

(in-ns 'clj-ml-wkg/aimes-house-prices)

;;Get past some dependency conflict

(def ds (load-dataset))

;;Either load initial results
(def gs-results (io/get-nippy "file://aimes-initial-results.nippy"))

;;Or train
(def gs-result (gridsearch-the-things))


;;I haven't put any real effort into nice viz.  HALP!

(accuracy-graph gs-results)


(->> gs-results
	 results->accuracy-dataset
	 (sort-by :average-loss)
	 (take 10))


({:average-loss 0.13688436523933248,
  :model-name ":libsvm/regression",
  :predict-time 5644,
  :train-time 26667}
 {:average-loss 0.13834159074948366,
  :model-name ":libsvm/regression",
  :predict-time 7252,
  :train-time 29334}
 {:average-loss 0.13866556643381417,
  :model-name ":libsvm/regression",
  :predict-time 5903,
  :train-time 25517}
 {:average-loss 0.13998699135016368,
  :model-name ":libsvm/regression",
  :predict-time 6692,
  :train-time 25583}
 {:average-loss 0.14019045847187636,
  :model-name ":libsvm/regression",
  :predict-time 7357,
  :train-time 30734}
 {:average-loss 0.1420648520842034,
  :model-name ":libsvm/regression",
  :predict-time 6479,
  :train-time 26369}
 {:average-loss 0.14248508919110822,
  :model-name ":libsvm/regression",
  :predict-time 5872,
  :train-time 25348}
 {:average-loss 0.1427344709806652,
  :model-name ":libsvm/regression",
  :predict-time 2878,
  :train-time 13883}
 {:average-loss 0.14287008716315316,
  :model-name ":libsvm/regression",
  :predict-time 6527,
  :train-time 25856}
 {:average-loss 0.14308160918576537,
  :model-name ":libsvm/regression",
  :predict-time 6295,
  :train-time 26868})
```

There is no real dataset engineering past what I had to do to get things to load.  It is interesting
that xgboost and svm beat everthing initially.  In my experience, this indicates that you need
to do feature engineering as for this type of problem the linear methods should dominate :-).


Note that I named us the `Clojure ML Working Group`.  This fits in my experience at least with
C++ the only groups that did anything useful were the working groups.

## License

Copyright © 2019 Clojure ML Working Group

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
