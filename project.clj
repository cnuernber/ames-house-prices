(defproject clj-ml-workgroup/ames-house-prices "0.1.0-SNAPSHOT"
  :description "Ames house prices kaggle comp"
  :url "http://github.com/clj-ml-workgroup"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.1-beta2"]
                 [techascent/tech.ml "1.0-alpha3"]
                 [metasoarous/oz "1.6.0-alpha2"]]
  :profiles {:dev {:dependencies [[org.clojure/tools.logging "0.3.1"]
                                  [ch.qos.logback/logback-classic "1.1.3"]]}})
