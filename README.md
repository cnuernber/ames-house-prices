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
