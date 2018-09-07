library(randomForest)
library(dplyr)
library(readr)
library(caret)
library(pROC)
setwd("~/Desktop/DSO530")
###Import Data

click= read.csv("smote_train.csv")
test = read_csv("hold_out_test.csv")


####Clean Data
click = click %>%
  select(-X) 

test = test %>%
  select(-X1) 

###Change to factor
click$is_attributed = as.factor(click$is_attributed)
test$is_attributed = as.factor(test$is_attributed)


###tune Mtry

set.seed(530)

t = tuneRF(click[,-43], click[,43],
                stepFactor = 2,
                plot = TRUE,
                ntreeTry = 400,
                trace = TRUE,
                improve = 0.05)
plot(rf)
legend("topright", colnames(rf$err.rate),col=1:4,fill=1:4)
###Build Model

rf = randomForest(is_attributed~.,data = click,mtry = 16, importance = TRUE, ntree =400)
print(rf)


###Feature Importance

importance(rf)
varImpPlot(rf, sort = T, n.var=15, main = "Top 15 Important Vars")


### prediction
pred = predict(rf,test)
confusionMatrix(pred, test$is_attributed)


###Stats

precision = posPredValue(pred, test$is_attributed, positive="1")
recall = sensitivity(pred, test$is_attributed, positive="1")
F1 = (2 * precision * recall) / (precision + recall)


###ROC curve
pred_Prob = predict(rf,test, type = "prob")
pred_Prob = as.data.frame(pred_Prob)


rf_roc = roc(test$is_attributed, pred_Prob$`1`, auc = T)
plot(rf_roc, print.auc=TRUE, auc.polygon=TRUE, grid = c(0.1,0.2),
     grid.col=c("green", "red"), max.auc.polygon = TRUE, auc.polygon.col = "skyblue",
     print.thres =TRUE)

