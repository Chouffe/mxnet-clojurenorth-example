(ns mxnet-cortex-example.core
  (:require
    [org.apache.clojure-mxnet.callback :as callback]
    [org.apache.clojure-mxnet.context :as context]
    [org.apache.clojure-mxnet.dtype :as d]
    [org.apache.clojure-mxnet.eval-metric :as eval-metric]
    [org.apache.clojure-mxnet.executor :as executor]
    [org.apache.clojure-mxnet.initializer :as initializer]
    [org.apache.clojure-mxnet.lr-scheduler :as lr-scheduler]
    [org.apache.clojure-mxnet.io :as mx-io]
    [org.apache.clojure-mxnet.module :as m]
    [org.apache.clojure-mxnet.util :as util]
    [org.apache.clojure-mxnet.ndarray :as ndarray]
    [org.apache.clojure-mxnet.optimizer :as optimizer]
    [org.apache.clojure-mxnet.shape :as mx-shape]
    [org.apache.clojure-mxnet.symbol :as sym]
    [org.apache.clojure-mxnet.visualization :as viz]))

;; Defining the computation graph of the Model
(defn get-symbol []
  (as-> (sym/variable "data") data
    (sym/fully-connected "fc1" {:data data :num-hidden 512})
    (sym/activation "act1" {:data data :act-type "relu"})
    (sym/dropout "drop1" {:data data :p 0.5})
    (sym/fully-connected "fc2" {:data data :num-hidden 128})
    (sym/activation "act2" {:data data :act-type "relu"})
    (sym/dropout "drop2" {:data data :p 0.5})
    (sym/fully-connected "fc3" {:data data :num-hidden 16})
    (sym/activation "act3" {:data data :act-type "relu"})
    (sym/fully-connected "fc4" {:data data :num-hidden 1})
    (sym/linear-regression-output "linear_regression" {:data data})))

;; Model Vizualisation

(defn render-model!
  "Render the `model-sym` and saves it as a png file in `path/model-name.png`
  using graphviz."
  [{:keys [model-name model-sym input-data-shape path]}]
  (let [dot (viz/plot-network
              model-sym
              {"data" input-data-shape}
              {:title model-name
               :node-attrs {:shape "oval" :fixedsize "false"}})]
    (viz/render dot "dot" "png" model-name path)))

;; Creating Datasets as NDArrays

(defn dataset->X-Y
  [dataset]
  (let [n-datapoints (count dataset)
        n-features (-> dataset first :features count)]
    {:X (ndarray/array (->> dataset (map :features) flatten (into []))
                       [n-datapoints n-features])
     :Y (ndarray/array (->> dataset (map :score) flatten (into []))
                       [n-datapoints 1])}))

(defn filename->dataset!
  [filename]
  (-> filename
      slurp
      read-string))

(defonce test-X-Y
  (-> "data/test.txt"
      filename->dataset!
      dataset->X-Y))

(defonce train-X-Y
  (-> "data/train.txt"
      filename->dataset!
      dataset->X-Y))

(def batch-size 2000)

(def test-iter
  (mx-io/ndarray-iter [(get test-X-Y :X)]
                      {:label-name "linear_regression_label"
                       :label [(get test-X-Y :Y)]
                       :data-batch-size batch-size}))

(def train-iter
  (mx-io/ndarray-iter [(get train-X-Y :X)]
                      {:label-name "linear_regression_label"
                       :label [(get train-X-Y :Y)]
                       :data-batch-size batch-size}))

(defn train!
  [model-mod train-iter test-iter num-epoch]
  (-> model-mod
      (m/bind {:data-shapes (mx-io/provide-data train-iter)
               :label-shapes (mx-io/provide-label test-iter)})
      (m/fit {:train-data train-iter
              :eval-data test-iter
              ;; Training for `num-epochs`
              :num-epoch num-epoch
              :fit-params
              (m/fit-params
                {:batch-end-callback (callback/speedometer batch-size 100)
                 ;; Initializing weights with Xavier
                 :initializer (initializer/xavier)
                 ;; Choosing Optimizer Algorithm: SGD with lr = 0.01
                 :optimizer
                 (optimizer/sgd
                   {:learning-rate 0.01
                    :momentum 0.001
                    :lr-scheduler (lr-scheduler/factor-scheduler 3000 0.9)})
                 ;; Evaluation Metric
                 :eval-metric (eval-metric/mse)})})))


(comment

  ;; Defining parameters
  (def train-epoch 100)
  (def load-epoch 10000)
  (def prefix "mymodel")
  (def model-name prefix)
  (def data-names ["data"])
  (def label-names ["linear_regression_label"])

  ;; Wrapping the computation graph in a `module`
  (def model-mod
    (m/module (get-symbol)
              {:data-names data-names
               :label-names label-names}))

  ; (sym/list-arguments (m/symbol model-mod))
  ; (sym/list-outputs (m/symbol model-mod))

  ;; Training the model for `epoch`
  (train! model-mod train-iter test-iter train-epoch)

  ;; Persisting model to disk
  (m/save-checkpoint model-mod
                     {:epoch train-epoch
                      :prefix prefix
                      :save-opt-state true})

  ;; Loading model from disk
  (def loaded-model-mod
    (-> {:prefix prefix
         :load-optimizer-states true
         :epoch load-epoch
         :data-names data-names
         :label-names label-names}
        ;; Loading from checkpoint
        (m/load-checkpoint)
        ;; Binding label and data shapes
        (m/bind {:data-shapes (mx-io/provide-data train-iter)
                 :label-shapes (mx-io/provide-label test-iter)})))

  ;;; Performace evaluation

  ;; From `loaded-model-mod`
  (m/score loaded-model-mod
           {:eval-data test-iter
            :eval-metric (eval-metric/mse)})

  ;; From `model-mod`
  (m/score model-mod
           {:eval-data test-iter
            :eval-metric (eval-metric/mse)})

  (defn sanity-check
    [k model-mod test-iter test-X-Y]
    (let [score (m/score model-mod
                         {:eval-data test-iter
                          :eval-metric (eval-metric/mse)})
          predictions (-> model-mod
                          (m/predict {:eval-data test-iter})
                          (first)
                          (ndarray/slice 0 k)
                          (ndarray/->vec))
          ground-truth (-> test-X-Y
                           :Y
                           (ndarray/slice 0 k)
                           (ndarray/->vec))]
      (println "Score on Test Set: ")
      (println score)
      (println "\nPredictions: ")
      (println predictions)
      (println "\nGround Truth: ")
      (println ground-truth)))

  (sanity-check 20 model-mod test-iter test-X-Y)
  ;Score on Test Set:
  ;[mse 1.9759508]
  ;
  ;Predictions:
  ;[2.8569365 3.0594287 2.947115 2.841632 3.2931566 3.0408115 2.9448495 2.993783 2.841546 2.54163 2.89647 3.3247905 2.6727371 3.1266866 2.9160218 3.3124614 2.988596 3.0056143 3.1018448 3.086196]
  ;
  ;Ground Truth:
  ;[3.5 4.0 2.0 5.0 3.0 4.5 1.0 1.5 5.0 0.5 5.0 5.0 5.0 3.0 5.0 5.0 0.5 2.5 3.0 5.0]

  (sanity-check 20 loaded-model-mod test-iter test-X-Y)
  ;Score on Test Set:
  ;[mse 1.5319124]
  ;
  ;Predictions:
  ;[3.0426044 4.099924 2.0988526 3.3540404 3.194833 3.372664 2.748798 3.8139641 2.934378 1.3496954 2.891445 3.1200047 2.1070197 3.511092 3.3848426 3.853949 1.9194205 3.2366083 3.2725945 2.3981385]
  ;
  ;Ground Truth:
  ;[3.5 4.0 2.0 5.0 3.0 4.5 1.0 1.5 5.0 0.5 5.0 5.0 5.0 3.0 5.0 5.0 0.5 2.5 3.0 5.0]

;; Vizualisation
  (render-model! {:model-name model-name
                  :model-sym (get-symbol)
                  :input-data-shape [1 203]
                  :path "model_render"}))
