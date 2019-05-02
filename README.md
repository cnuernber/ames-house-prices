# ames-house-prices

Exploration of kaggle ames house prices comp.

Check out the [nbviewer version](https://nbviewer.jupyter.org/github/cnuernber/ames-house-prices/blob/82e3ce1679b3e6e31c0128290f60ef7ae16947b0/ames-housing-prices-clojure.ipynb).

## Usage

From top level directory:
```
scripts/get-data.sh
```

The data is under data/ames-house-prices


Make sure you have openblas or atlas installed as well as libsvm.

If SVM doesn't work just remove it from the gridsearch pathways.  It is an old C
library that can be temperamental.  From ubuntu it works fine on jvm 8.


There is some dependency conflict with the csv subsystem tablesaw so from here is the workflow:

```clojure
lein repl

;;load the ames namespace
(require '[clj-ml-wkg/ames-house-prices])

(in-ns 'clj-ml-wkg/ames-house-prices)

;; To see progress against reference article: 

(presentation)

```

## License

Copyright © 2019 Clojure ML Working Group

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
