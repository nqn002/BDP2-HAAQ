import json
import math
import io
import os
import pickle
import sys
import yaml
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import tree
from dvclive import Live
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_score, recall_score

def evaluate(model, matrix, labels):
    
    # Configure file path
    #file_path = os.path.join(evaluation_dir, file_name)

	# Predicted labels
    prediction_prob = model.predict(matrix)
    predictions = np.round(abs(prediction_prob))
    loss_and_accuracy = model.evaluate(test_matrix,Y_test)
    
    # Log plots
    live.log_plot("roc", labels.astype(float), predictions.astype(float))
    live.log_plot("confusion_matrix", labels.squeeze(), predictions.argmax(-1))
    
    # Log average metrics
    live.log("avg_prec", metrics.average_precision_score(labels, predictions))
    live.log("roc_auc", metrics.roc_auc_score(labels, predictions))
    live.log("loss",loss_and_accuracy[0])
    live.log("accuracy",loss_and_accuracy[1])

    # Log precision and recall 
    precision, recall, prc_thresholds = metrics.precision_recall_curve(labels, predictions)
    nth_point = math.ceil(len(prc_thresholds) / 1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]
    
    prc_file = os.path.join(evaluation_dir, "plots", "precision_recall.json")
    
    with open(prc_file, "w") as fd:
        json.dump(
            {
                "prc": [
                    {"precision": p.astype(float), "recall": r.astype(float), "threshhold": t.astype(float)}
                    for p, r, t in prc_points
                ]
            },
            fd,
            indent=4,
        )
    
    return None
    
    
    
############################ Mainline proessing starts here  ############################ 

params = yaml.safe_load(open("params.yaml"))["evaluate"]
average = params['average']

if len(sys.argv) != 4:
    len_sys = len(sys.argv)
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython evaluate.py \n")
    sys.stderr.write("\tExpected {} input arguments\n".format(str(4)))
    sys.stderr.write("\tReceived {} arguments\n".format(str(len_sys)))
    sys.exit(1)

# Read input
model_input = sys.argv[1]
with open(model_input, "rb") as fd:
    model = pickle.load(fd)

# Read train data for evaluation
train_input = sys.argv[2]
with open(train_input, "rb") as fd:
    train_matrix, Y_train, train_max_len, train_max_words = pickle.load(fd)

# Read test data for evaluation
test_input = sys.argv[3]
with open(test_input, "rb") as fd:
    test_matrix, Y_test, test_max_len, test_max_words = pickle.load(fd)
  
# Track live evaluation
evaluation_dir = "evaluation"
live = Live(evaluation_dir)

# Evaluate train dataset
#evaluate(model, train_matrix, Y_train, "train")

# Evaluate test dataset
evaluate(model, test_matrix, Y_test)
