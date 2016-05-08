import pandas as pd
from sklearn import metrics

def calcROC(ypred,ytrue):
   fpr, tpr, _ = metrics.roc_curve(ytrue, ypred)
   df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
   auc = metrics.auc(fpr,tpr)
   print "ROC Curve w/ AUC=%0.3f" %auc
   return auc



