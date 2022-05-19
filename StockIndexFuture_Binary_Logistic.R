#####Library#####
{
  library(gbm)
  library(readxl)
  library(MASS)
  library(lmtest)
  library(sandwich)
  library(mfx)
  library(caret)
  library(ROCR)
  library(woe)
  library(doParallel)
  library(DAAG)
  library(boot)
  library(dplyr)
}

#####Setting#####
setwd("C:/RRR/")
stockfuture <- read_excel("完整資料.xlsx")

###如果設成factor跑gbm，R會當掉
stockfuture$positive_revenue <- as.factor(ifelse(stockfuture$nextday_openchange >= 0,"1","0"))

summary(stockfuture)
#從summary來看，58%是正報酬，42%是負報酬，接近1:1

stockfuture_train <- stockfuture[1:351,]
stockfuture_test  <- stockfuture[352:475,]

#####Without Feature Selelction#####

stockfuture_probit <- glm(positive_revenue ~ . - date - nextday_openchange,
                          data = stockfuture_train , family = binomial(probit)
)
summary(stockfuture_probit)
#glm(positive_revenue ~ VIXchange + SP500change + VIXvolume + bitchange, data = stockfuture_train , family = binomial(probit))
#iv.mult(data.frame(stockfuture_train[,3:ncol(stockfuture_train)]) , y = "positive_revenue" , summary =TRUE , verbose = FALSE)
#iv.mult(data.frame(stockfuture_train[,3:36]) , y = "positive_revenue" , summary =TRUE , verbose = FALSE)
#probit可能遇到algorithm did not converge的問題，可能某些變數的線性組合能完美的預測漲跌
#可以看到SP500和VIX的變動可能可以完全的預測台指期走勢方向
#這邊改用logit試試看
stockfuture_logit <- glm(positive_revenue ~ . - date - nextday_openchange , data = stockfuture_train , family = binomial("logit"))
summary(stockfuture_logit)
#coeftest(stockfuture_logit, vcov. = vcovHC, type = "HC3")
#logLik(stockfuture_logit)
#logitmfx(positive_revenue ~ . - date - nextday_openchange , data = stockfuture_train, atmean = FALSE) #APE instead of PEA

#觀察預測結果
sum(stockfuture_train$positive_revenue=="1")
#而因為在train裡面的陽性(正報酬)占比為213/351 ~= 0.606 ，我們這邊以0.6的log勝算比當作門檻值
stockfuture.binary.logit.pred.train_raw <- predict( stockfuture_logit , newdata = stockfuture_train)
stockfuture.binary.logit.pred.train <- ifelse(stockfuture.binary.logit.pred.train_raw >= log(0.6/0.4) , 1,0)

stockfuture.binary.logit.pred.test_raw <- predict( stockfuture_logit , newdata = stockfuture_test)
stockfuture.binary.logit.pred.test <- ifelse(stockfuture.binary.logit.pred.test_raw >= log(0.6/0.4) , 1,0)

confusionMatrix(data = factor(stockfuture.binary.logit.pred.train) , reference = factor(stockfuture_train$positive_revenue),positive = "1")
confusionMatrix(data = factor(stockfuture.binary.logit.pred.test) , reference = factor(stockfuture_test$positive_revenue),positive = "1")
#logit樣本外Accuracy為0.7661
#TPR(true positive rate)為0.7164，TNR(true negative rate)為0.8246
#PPV(positive predictive value)為0.8276，NPV(negative predictive value)為0.7121

pred_logit_train <- prediction(stockfuture.binary.logit.pred.train_raw , stockfuture_train$positive_revenue) 
perf_logit_train <-performance (pred_logit_train , measure = "tpr" , x.measure = "fpr")
performance(pred_logit_train, measure = "auc")@y.values[[1]]
#樣本內AUC為0.860

pred_logit <- prediction(stockfuture.binary.logit.pred.test_raw , stockfuture_test$positive_revenue) 
perf_logit <-performance (pred_logit , measure = "tpr" , x.measure = "fpr")
performance(pred_logit, measure = "auc")@y.values[[1]]
#logit樣本外AUC為0.844


######PCA#####
PCA <- read.csv("PCA.csv")
stockfuture_PCA <- cbind( positive_revenue =  stockfuture$positive_revenue , PCA)
stockfuture_PCA$positive_revenue <- as.factor(ifelse(stockfuture$nextday_openchange >= 0,1,0))

stockfuture_PCA_train <- stockfuture_PCA[1:351,]
stockfuture_PCA_test  <- stockfuture_PCA[352:475,]

stockfuture_PCA_logit <- glm(positive_revenue ~ .  , data = stockfuture_PCA_train , family = binomial("logit"))
summary(stockfuture_PCA_logit)

stockfuture.PCA.logit.pred.train_raw <- predict( stockfuture_PCA_logit , newdata = stockfuture_PCA_train)
stockfuture.PCA.logit.pred.train <- ifelse(stockfuture.PCA.logit.pred.train_raw >= log(0.6/0.4) , 1,0)
confusionMatrix(data = factor(stockfuture.PCA.logit.pred.train) , reference = factor(stockfuture_PCA_train$positive_revenue),positive = "1")
#樣本內Accuracy為0.698

stockfuture.PCA.logit.pred.test_raw <- predict( stockfuture_PCA_logit , newdata = stockfuture_PCA_test)
stockfuture.PCA.logit.pred.test <- ifelse(stockfuture.PCA.logit.pred.test_raw >= log(0.6/0.4) , 1,0)
confusionMatrix(data = factor(stockfuture.PCA.logit.pred.test) , reference = factor(stockfuture_PCA_test$positive_revenue),positive = "1")
#PCA logit的樣本外Accuracy為0.645

pred_PCA_logit_train <- prediction(stockfuture.PCA.logit.pred.train_raw , stockfuture_PCA_train$positive_revenue) 
perf_PCA_logit_train <-performance (pred_PCA_logit_train , measure = "tpr" , x.measure = "fpr")
performance(pred_PCA_logit_train, measure = "auc")@y.values[[1]]
#樣本內AUC為0.753

pred_PCA_logit <- prediction(stockfuture.PCA.logit.pred.test_raw , stockfuture_PCA_test$positive_revenue) 
perf_PCA_logit <-performance (pred_PCA_logit , measure = "tpr" , x.measure = "fpr")
performance(pred_PCA_logit, measure = "auc")@y.values[[1]]
#PCA logit的AUC為0.733


#####LASSO#####
stockfuture %>%
  dplyr::select(positive_revenue,SP500change,SP500change_lag2,VIXvolume,USDchange,VIXchange,bitchange) -> stockfuture_LASSO

stockfuture_LASSO_train <- stockfuture_LASSO[1:351,]
stockfuture_LASSO_test  <- stockfuture_LASSO[352:475,]

stockfuture_LASSO_logit <- glm(positive_revenue ~ .  , data = stockfuture_LASSO_train , family = binomial("logit"))
summary(stockfuture_LASSO_logit)

stockfuture.LASSO.logit.pred.train_raw <- predict( stockfuture_LASSO_logit , newdata = stockfuture_LASSO_train)
stockfuture.LASSO.logit.pred.train <- ifelse(stockfuture.LASSO.logit.pred.train_raw >= log(0.6/0.4) , 1,0)
confusionMatrix(data = factor(stockfuture.LASSO.logit.pred.train) , reference = factor(stockfuture_LASSO_train$positive_revenue),positive = "1")
#樣本內Accuracy為0.755

stockfuture.LASSO.logit.pred.test_raw <- predict( stockfuture_LASSO_logit , newdata = stockfuture_LASSO_test)
stockfuture.LASSO.logit.pred.test <- ifelse(stockfuture.LASSO.logit.pred.test_raw >= log(0.6/0.4) , 1,0)
confusionMatrix(data = factor(stockfuture.LASSO.logit.pred.test) , reference = factor(stockfuture_LASSO_test$positive_revenue),positive = "1")
#LASSO logit的樣本外Accuracy為0.7823

pred_LASSO_logit_train <- prediction(stockfuture.LASSO.logit.pred.train_raw , stockfuture_LASSO_train$positive_revenue) 
perf_LASSO_logit_train <-performance (pred_LASSO_logit_train , measure = "tpr" , x.measure = "fpr")
performance(pred_LASSO_logit_train, measure = "auc")@y.values[[1]]
#樣本內AUC為0.817

pred_LASSO_logit <- prediction(stockfuture.LASSO.logit.pred.test_raw , stockfuture_LASSO_test$positive_revenue) 
perf_LASSO_logit <-performance (pred_LASSO_logit , measure = "tpr" , x.measure = "fpr")
performance(pred_LASSO_logit, measure = "auc")@y.values[[1]]
#LASSO logit的樣本外AUC為0.847


#####RF#####
stockfuture %>%
  dplyr::select(positive_revenue,VIXchange,SP500change,SP500change_lag1,VIXvolume,ETHchange,openchange_lag1) -> stockfuture_RF

stockfuture_RF_train <- stockfuture_RF[1:351,]
stockfuture_RF_test  <- stockfuture_RF[352:475,]

stockfuture_RF_logit <- glm(positive_revenue ~ .  , data = stockfuture_RF_train , family = binomial("logit"))
summary(stockfuture_RF_logit)

stockfuture.RF.logit.pred.train_raw <- predict( stockfuture_RF_logit , newdata = stockfuture_RF_train)
stockfuture.RF.logit.pred.train <- ifelse(stockfuture.RF.logit.pred.train_raw >= log(0.6/0.4) , 1,0)
confusionMatrix(data = factor(stockfuture.RF.logit.pred.train) , reference = factor(stockfuture_RF_train$positive_revenue),positive = "1")
#樣本內Accuracy為0.749

stockfuture.RF.logit.pred.test_raw <- predict( stockfuture_RF_logit , newdata = stockfuture_RF_test)
stockfuture.RF.logit.pred.test <- ifelse(stockfuture.RF.logit.pred.test_raw >= log(0.6/0.4) , 1,0)
confusionMatrix(data = factor(stockfuture.RF.logit.pred.test) , reference = factor(stockfuture_RF_test$positive_revenue),positive = "1")
#RF logit的樣本外Accuracy為0.750

pred_RF_logit_train <- prediction(stockfuture.RF.logit.pred.train_raw , stockfuture_RF_train$positive_revenue) 
perf_RF_logit_train <-performance (pred_RF_logit_train , measure = "tpr" , x.measure = "fpr")
performance(pred_RF_logit_train, measure = "auc")@y.values[[1]]
#樣本內AUC為0.798

pred_RF_logit <- prediction(stockfuture.RF.logit.pred.test_raw , stockfuture_RF_test$positive_revenue) 
perf_RF_logit <-performance (pred_RF_logit , measure = "tpr" , x.measure = "fpr")
performance(pred_RF_logit, measure = "auc")@y.values[[1]]
#RF logit的樣本外AUC為0.820


#####Core#####

stockfuture %>%
  dplyr::select(positive_revenue,VIXchange,SP500change,VIXvolume) -> stockfuture_core

stockfuture_core_train <- stockfuture_core[1:351,]
stockfuture_core_test  <- stockfuture_core[352:475,]

stockfuture_core_logit <- glm(positive_revenue ~ .  , data = stockfuture_core_train , family = binomial("logit"))
summary(stockfuture_core_logit)

stockfuture.core.logit.pred.train_raw <- predict( stockfuture_core_logit , newdata = stockfuture_core_train)
stockfuture.core.logit.pred.train <- ifelse(stockfuture.core.logit.pred.train_raw >= log(0.6/0.4) , 1,0)
confusionMatrix(data = factor(stockfuture.core.logit.pred.train) , reference = factor(stockfuture_core_train$positive_revenue),positive = "1")
#樣本內Accuracy為0.7521

stockfuture.core.logit.pred.test_raw <- predict( stockfuture_core_logit , newdata = stockfuture_core_test)
stockfuture.core.logit.pred.test <- ifelse(stockfuture.core.logit.pred.test_raw >= log(0.6/0.4) , 1,0)
confusionMatrix(data = factor(stockfuture.core.logit.pred.test) , reference = factor(stockfuture_core_test$positive_revenue),positive = "1")
#core logit的樣本外Accuracy為0.7903

pred_core_logit_train <- prediction(stockfuture.core.logit.pred.train_raw , stockfuture_core_train$positive_revenue) 
perf_core_logit_train <-performance (pred_core_logit_train , measure = "tpr" , x.measure = "fpr")
performance(pred_core_logit_train, measure = "auc")@y.values[[1]]
#樣本內AUC為0.796

pred_core_logit <- prediction(stockfuture.core.logit.pred.test_raw , stockfuture_core_test$positive_revenue) 
perf_core_logit <-performance (pred_core_logit , measure = "tpr" , x.measure = "fpr")
performance(pred_core_logit, measure = "auc")@y.values[[1]]
#core logit的樣本外AUC為0.828