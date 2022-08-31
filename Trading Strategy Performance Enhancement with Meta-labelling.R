# SET UP ------------------------------------------------------------------
install.packages("glue")
require("glue")
require(rpart)
install.packages("rpart.plot")
library("rpart.plot")

install.packages("nnet")
require("nnet")

library(adabag)

require(randomForest)
install.packages("gbm")
require(gbm)

dfEQ = read.csv("C:/Users/j_sto/Documents/MSc Applied Statistics and Financial Modelling/Final Year Project/Futures Markets/Commodity Futures/EQ_Portfolio Returns & Macro Data.csv",header=TRUE)
dfEQ$total <- as.factor(dfEQ$total)

splitEQ <- c(rep(0, round(0.7 * nrow(dfEQ),0)), rep(1, round(0.3 * nrow(dfEQ),0)))
trainEQ = dfEQ[splitEQ == 0,]
testEQ = dfEQ[splitEQ == 1,]

# Useful Functions --------------------------------------------------------

#In sample performance metrics
in_sample_performance <- function(tree_object) {
  tree_object_pred = predict(tree_object,newdata = subset(trainEQ, select=c(-month) ),type="class")
  table <- table(trainEQ$total, tree_object_pred)
  tree_train_accuracy = sum(diag(table)) / sum(table)
  tree_train_precision = table[4] / sum(table[3:4])
  tree_train_recall = table[4] / sum(table[2,1:2])
  tree_train_F = (2*tree_train_precision*tree_train_recall)/(tree_train_precision+tree_train_recall)
  tree_train_specificity = table[1] / sum(table[1,1:2])
  
  Metric = c("Accuracy","Precision","Recall","Specificity","F1")
  Value = c(tree_train_accuracy,tree_train_precision,tree_train_recall,tree_train_specificity,tree_train_F)
  data_frame = data.frame(Metric, Value)
  return(data_frame)
}

#Out of sample performance metrics
out_of_sample_performance <- function(tree_object,Type) {
  tree_object_pred = predict(tree_object,newdata = subset(testEQ, select=c(-month) ),type="class")
  table <- table(testEQ$total, tree_object_pred)
  tree_train_accuracy = sum(diag(table)) / sum(table)
  tree_train_precision = table[4] / sum(table[3:4])
  tree_train_recall = table[4] / sum(table[2,1:2])
  tree_train_F = (2*tree_train_precision*tree_train_recall)/(tree_train_precision+tree_train_recall)
  tree_train_specificity = table[1] / sum(table[1,1:2])
  
  Metric = c("Accuracy","Precision","Recall","Specificity","F1")
  Value = c(tree_train_accuracy,tree_train_precision,tree_train_recall,tree_train_specificity,tree_train_F)
  data_frame = data.frame(Metric, Value)
  names(data_frame)[names(data_frame) == 'Value'] <- Type
  return(data_frame)
}

out_of_sample_performanceAB <- function(tree_object,Type) {
  predictions_accAdaBoost <- predict(tree_object,newdata = subset(testEQ, select=c(-month) ),type="class")
  predictionsEQ_AB <- predictions_accAdaBoost$class
  table <- table(testEQ$total, predictionsEQ_AB)
  tree_train_accuracy = sum(diag(table)) / sum(table)
  tree_train_precision = table[4] / sum(table[3:4])
  tree_train_recall = table[4] / sum(table[2,1:2])
  tree_train_F = (2*tree_train_precision*tree_train_recall)/(tree_train_precision+tree_train_recall)
  tree_train_specificity = table[1] / sum(table[1,1:2])
  
  Metric = c("Accuracy","Precision","Recall","Specificity","F1")
  Value = c(tree_train_accuracy,tree_train_precision,tree_train_recall,tree_train_specificity,tree_train_F)
  data_frame = data.frame(Metric, Value)
  names(data_frame)[names(data_frame) == 'Value'] <- Type
  return(data_frame)
}



#Exporting for Python script
in_sample_python_export <- function(tree_object,type){
  predictionsEQ <- predict(tree_object,newdata = subset(trainEQ, select=c(-month ) ),type="prob")
  predictionsEQ <- data.frame(predictionsEQ)
  colnames(predictionsEQ) <- c(0, 1)
  predictionsEQ <- merge(predictionsEQ, dfEQ$month, by.x=0, by.y=0,sort = FALSE)
  a = "In Sample_Tree Predictions_"
  write.csv(predictionsEQ,glue("{a}{type}.csv"), row.names = TRUE)
}

out_of_sample_python_export <- function(tree_object,type){
  predictionsEQ <- predict(tree_object,newdata = subset(testEQ, select=c(-month ) ),type="prob")
  predictionsEQ <- data.frame(predictionsEQ)
  colnames(predictionsEQ) <- c(0, 1)
  predictionsEQ <- merge(predictionsEQ, dfEQ$month, by.x=0, by.y=0,sort = FALSE)
  a = "Tree Predictions_"
  write.csv(predictionsEQ,glue("{a}{type}.csv"), row.names = TRUE)
}

# SINGLE CLASSIFICATION TREE -----------------------------------------------------------

loss_matrix1 = matrix(c(0,2,1,0), byrow=TRUE, nrow=2) # false positive twice as bad as false negative 

#Fitting with Loss Matrix - 10 fold CV
set.seed(666)
library(rpart)
treeEQ1 <- rpart(total ~ . , data=subset(trainEQ, select=c(-month) ),parms=list(loss=loss_matrix1), method="class",cp=0.001)
printcp(treeEQ1)

#export to Python
out_of_sample_python_export(treeEQ1,"EQ_Tree")

#out-of-sample performance
out_of_sample_performance(treeEQ1,"Tree")

plotcp(treeEQ1)

rpart.plot(treeEQ1)


# RANDOM FOREST -----------------------------------------------------------

set.seed(666)
rfEQ <- randomForest(total ~ . , data=subset(trainEQ, select=c(-month)), method="class")
rfEQ

set.seed(666)
rfEQ <- randomForest(total ~ . , data=subset(trainEQ, select=c(-month)), method="class", ntree = 500, mtry = 9) #increase variables used to reduce oob error rate
rfEQ

#Export to Python
predictionsEQ_RF <- predict(rfEQ,newdata = subset(testEQ, select=c(-month ) ),type="prob")
predictionsEQ_RF <- data.frame(predictionsEQ_RF)
colnames(predictionsEQ_RF) <- c(0, 1)
predictionsEQ_RF <- merge(predictionsEQ_RF, dfEQ$month, by.x=0, by.y=0,sort = FALSE)
write.csv(predictionsEQ_RF,"Tree Predictions_EQ_RF.csv", row.names = TRUE)

#out-of-sample performance
out_of_sample_performance(rfEQ,"Random Forest")

#Importance plot
par(cex=0.5)
varImpPlot(rfEQ, main="Random Forest - Variable Importance Plot", n.var=40)

# ADABOOST ----------------------------------------------------------------


set.seed(666)
AdaBoostEQ <- boosting(total ~ . , data=subset(trainEQ, select=c(-month)), boos = TRUE, mfinal = 100, coeflearn = "Breiman")

#Below shows that tuning the number of iterations with CV is not that helpful
set.seed(666)
AdaBoostEQ_CV <- boosting.cv(total ~ . , data=subset(trainEQ, select=c(-month)), boos = TRUE, v=10, mfinal = 100, coeflearn = "Breiman")
AdaBoostEQ_CV$error

predictions_accAdaBoost <- predict(AdaBoostEQ,newdata = subset(testEQ, select=c(-month) ),type="class")

#Export to Python
predictions_accAdaBoost <- predict(AdaBoostEQ,newdata = subset(testEQ, select=c(-month) ),type="class")
predictionsEQ_AB <- predictions_accAdaBoost$prob
predictionsEQ_AB <- data.frame(predictionsEQ_AB)
colnames(predictionsEQ_AB) <- c(0, 1)
rownames(predictionsEQ_AB) <- predictionsEQ$Row.names
predictionsEQ_AB <- merge(predictionsEQ_AB, dfEQ$month, by.x=0, by.y=0,sort = FALSE)
write.csv(predictionsEQ_AB,"Tree Predictions_EQ_AB.csv", row.names = TRUE)

#Performance metrics
out_of_sample_performanceAB(AdaBoostEQ,"AdaBoost")

#Importance Plot
par(mar=c(5, 10, 5, 5) + 0.5)
importanceplot(AdaBoostEQ, horiz=TRUE, cex.names=.6)

# Breiman Bagging ---------------------------------------------------------

set.seed(666)
BagEQ <- bagging(total ~ ., data=subset(trainEQ, select=c(-month) ), v = 10, mfinal = 100)
printcp(BagEQ)

predictions_accBag <- predict(BagEQ,newdata = subset(testEQ, select=c(-month) ),type="class")

#Export to Python
predictionsEQ_Bag <- predictions_accBag$prob
predictionsEQ_Bag <- data.frame(predictionsEQ_Bag)
colnames(predictionsEQ_Bag) <- c(0, 1)
rownames(predictionsEQ_Bag) <- predictionsEQ$Row.names
predictionsEQ_Bag <- merge(predictionsEQ_Bag, dfEQ$month, by.x=0, by.y=0,sort = FALSE)
write.csv(predictionsEQ_Bag,"Tree Predictions_EQ_Bag.csv", row.names = TRUE)

#Performance metrics
out_of_sample_performanceAB(BagEQ,"Bagging")

#Importance plot
importanceplot(BagEQ, horiz=TRUE, cex.names=.6)


