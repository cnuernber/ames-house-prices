(defproject clj-ml-workgroup/ames-house-prices "0.1.0-SNAPSHOT"
  :description "Ames house prices kaggle comp"
  :url "http://github.com/clj-ml-workgroup"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [techascent/tech.ml "0.25"]
                 [techascent/tech.lentils "0.02"]
                 [metasoarous/oz "1.5.2"]]
  :profiles {:dev {:dependencies [[org.clojure/tools.logging "0.3.1"]
                                  [ch.qos.logback/logback-classic "1.1.3"]]}})
