from sklearn.metrics import confusion_matrix

def confusionMatrix(yTarget,yPrediction):
   return confusion_matrix(yTarget, yPrediction)

