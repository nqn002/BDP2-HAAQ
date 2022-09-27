# Adding new library for evaluation 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Modelling in train.py

#Evaluation 
Y_pred= model.predict(test_sequences_matrix)
confusion_matrix(Y_test, np.round(abs(Y_pred)), labels=[0,1])
print(classification_report(Y_test, np.round(abs(Y_pred))))
roc_auc_score(Y_test, Y_pred)