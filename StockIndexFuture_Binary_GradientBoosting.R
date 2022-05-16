#####Setting#####
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

stockfuture <- read_excel("C:/RRR/完整資料.xlsx")

###如果設成factor跑gbm，R會當掉
stockfuture$positive_revenue <- as.factor(ifelse(stockfuture$nextday_openchange >= 0,"1","0"))

summary(stockfuture)
#從summary來看，58%是正報酬，42%是負報酬，接近1:1

stockfuture_train <- stockfuture[1:351,]
stockfuture_test  <- stockfuture[352:475,]

#####Without feature selection#####

#####logit#####

stockfuture_probit <- glm(positive_revenue ~ . - date - nextday_openchange, #- VIXchange - SP500change,
                          data = stockfuture_train , family = binomial(probit)
                          )
summary(stockfuture_probit)
#glm(positive_revenue ~ VIXchange + SP500change + VIXvolume + bitchange, data = stockfuture_train , family = binomial(probit))
iv.mult(data.frame(stockfuture_train[,3:ncol(stockfuture_train)]) , y = "positive_revenue" , summary =TRUE , verbose = FALSE)
#iv.mult(data.frame(stockfuture_train[,3:36]) , y = "positive_revenue" , summary =TRUE , verbose = FALSE)
#probit可能遇到algorithm did not converge的問題，可能某些變數的線性組合能完美的預測漲跌
#可以看到SP500和VIX的變動可能可以完全的預測台指期走勢方向
#這邊改用logit試試看
stockfuture_logit <- glm(positive_revenue ~ . - date - nextday_openchange , data = stockfuture_train , family = binomial("logit"))
summary(stockfuture_logit)
coeftest(stockfuture_logit, vcov. = vcovHC, type = "HC3")
logLik(stockfuture_logit)
logitmfx(positive_revenue ~ . - date - nextday_openchange , data = stockfuture_train, atmean = FALSE) #APE instead of PEA

#觀察預測結果
sum(stockfuture_train$positive_revenue=="1")
#而因為在train裡面的陽性(正報酬)占比為213/351 ~= 0.606 ，我們這邊以0.6的log勝算比當作門檻值
stockfuture.binary.logit.pred.train_raw <- predict( stockfuture_logit , newdata = stockfuture_train)
stockfuture.binary.logit.pred.train <- ifelse(stockfuture.binary.logit.pred.train_raw >= log(0.6/0.4) , 1,0)

stockfuture.binary.logit.pred.test_raw <- predict( stockfuture_logit , newdata = stockfuture_test)
stockfuture.binary.logit.pred.test <- ifelse(stockfuture.binary.logit.pred.test_raw >= log(0.6/0.4) , 1,0)

confusionMatrix(data = factor(stockfuture.binary.logit.pred.train) , reference = factor(stockfuture_train$positive_revenue),positive = "1")
confusionMatrix(data = factor(stockfuture.binary.logit.pred.test) , reference = factor(stockfuture_test$positive_revenue),positive = "1")
#Accuracy為0.7661
#TPR(true positive rate)為0.7164，TNR(true negative rate)為0.8246
#PPV(positive predictive value)為0.8276，NPV(negative predictive value)為0.7121

pred_logit <- prediction(stockfuture.binary.logit.pred.test_raw , stockfuture_test$positive_revenue) 
perf_logit <-performance (pred_logit , measure = "tpr" , x.measure = "fpr")
performance(pred_logit, measure = "auc")@y.values[[1]]
#AUC為0.844


#####Gradient Boosting#####

#Tuning
ctrl <- trainControl(method = "repeatedcv",number = 10, repeats = 2, allowParallel = T , verboseIter = TRUE) #重複的cv，每次分10組，重複2次
registerDoParallel(detectCores()-1)
grid <- expand.grid(n.trees = c(2500,5000), interaction.depth=c(1:2), shrinkage=c(0.01,0.005,0.001) , n.minobsinnode=c(40,50,60))

#grid <- expand.grid(n.trees = c(5000,10000), interaction.depth=c(1), shrinkage=c(0.001) , n.minobsinnode=c(5,10,20))
set.seed(1)
#stockfuture.gbm.caret <- caret::train(positive_revenue ~ . - date - nextday_openchange, data = stockfuture_train, method = "gbm", metric = "Accuracy",
#                                      trControl = ctrl, tuneGrid = grid)
#print(stockfuture.gbm.caret)
#n.tree1000時表現不好，超過5000之後準確度下滑，可以考慮interaction.depth為2或1
#n.minobsinnode稍微大點表現越好
#最後決定n.minobsinnode=40，interaction.depth=1(or 2)，shrinkage=0.001，n.tree=5000

#build model
stockfuture$positive_revenue <- ifelse(stockfuture$nextday_openchange >= 0,1,0)
stockfuture_train <- stockfuture[1:351,]
stockfuture_test  <- stockfuture[352:475,]

set.seed(1)
stockfuture_binary_gbm <- gbm(positive_revenue ~ . - date - nextday_openchange , data = stockfuture_train,
                         distribution = "bernoulli", n.trees = 5000,
                         interaction.depth = 1, shrinkage = 0.001,n.minobsinnode = 40,
                         bag.fraction = 0.5, cv.folds = 10)

summary(stockfuture_binary_gbm)


stockfuture.binary.gbm.pred.train_raw <- predict.gbm( object = stockfuture_binary_gbm , newdata = stockfuture_train)
stockfuture.binary.gbm.pred.train <- ifelse(stockfuture.binary.gbm.pred.train_raw >= log(0.6/0.4) , 1,0)

stockfuture.binary.gbm.pred.test_raw <- predict.gbm( object = stockfuture_binary_gbm , newdata = stockfuture_test)
stockfuture.binary.gbm.pred.test <- ifelse(stockfuture.binary.gbm.pred.test_raw >= log(0.6/0.4) , 1,0) 

#我們用的是Bernoulli Loss，他的回傳值是Log odds，因此若pred>0則為正，pred<0則為負
#而因為在train裡面的陽性(正報酬)占比為0.6，我們這邊以0.6的log勝算比當作門檻值

confusionMatrix(data = factor(stockfuture.binary.gbm.pred.train) , reference = factor(stockfuture_train$positive_revenue),positive = "1")
confusionMatrix(data = factor(stockfuture.binary.gbm.pred.test) , reference = factor(stockfuture_test$positive_revenue),positive = "1")
#Test Data的Accuracy為 0.7742 ，Sensitivity為0.8955，Specificity為0.6316
#從實際結果來看，如果實際報酬為正，我們很高機率可以預測正確(0.8806)，如果實際報酬為負，我們只有較低機率預測正確(0.6491)
#從預測的值來看，如果我們預測報酬為正，那預測正確機率為0.7407，如果我們預測報酬為負，那預測正確的機率為0.8372，
#或著是說TPR(true positive rate)為0.8806，TNR(true negative rate)為0.6316
#以及PPV(positive predictive value)為0.7407，NPV(negative predictive value)為0.8372


#畫ROC圖
pred_logit <- prediction(stockfuture.binary.logit.pred.test_raw , stockfuture_test$positive_revenue) 
perf_logit <-performance (pred_logit , measure = "tpr" , x.measure = "fpr")
performance(pred_logit, measure = "auc")@y.values[[1]]

pred_GB <- prediction(stockfuture.binary.gbm.pred.test_raw , stockfuture_test$positive_revenue) 
perf_GB <-performance (pred_GB , measure = "tpr" , x.measure = "fpr")
performance(pred_GB, measure = "auc")@y.values[[1]]
#Gradient Boosting的AUC為0.867

windows()
plot(perf_GB , col = "red" , main = "ROC curve" , xlab = "1-Specificity (FPR)" , ylab = "Sensitivity(TPR)" , lwd = 5)
plot(perf_logit , col = "blue" , lwd = 5 , add = TRUE)
abline(0, 1 , lwd = 2)
legend("bottomright",legend = c("GradientBoosting(AUC = 0.867)","Logistic(AUC = 0.844)"), lty = 1 , col = c("red","blue") , lwd = 5)

#####After feature selection#####

######PCA#####
PCA <- read.csv("C:/RRR/PCA.csv")
stockfuture_PCA <- cbind( positive_revenue =  stockfuture$positive_revenue , PCA)

stockfuture_PCA_train <- stockfuture_PCA[1:351,]
stockfuture_PCA_test  <- stockfuture_PCA[352:475,]

#logit
stockfuture_PCA_logit <- glm(positive_revenue ~ .  , data = stockfuture_PCA_train , family = binomial("logit"))
summary(stockfuture_PCA_logit)

stockfuture.PCA.logit.pred.test_raw <- predict( stockfuture_PCA_logit , newdata = stockfuture_PCA_test)
stockfuture.PCA.logit.pred.test <- ifelse(stockfuture.PCA.logit.pred.test_raw >= log(0.6/0.4) , 1,0)
confusionMatrix(data = factor(stockfuture.PCA.logit.pred.test) , reference = factor(stockfuture_PCA_test$positive_revenue),positive = "1")
#PCA logit的Accuracy為0.645

pred_PCA_logit <- prediction(stockfuture.PCA.logit.pred.test_raw , stockfuture_PCA_test$positive_revenue) 
perf_PCA_logit <-performance (pred_PCA_logit , measure = "tpr" , x.measure = "fpr")
performance(pred_GB, measure = "auc")@y.values[[1]]
#PCA logit的AUC為0.733

#Gradient Boosting
stockfuture_PCA$positive_revenue <- ifelse(stockfuture$nextday_openchange >= 0,1,0)
stockfuture_PCA_train <- stockfuture_PCA[1:351,]
stockfuture_PCA_test  <- stockfuture_PCA[352:475,]

set.seed(1)
stockfuture_PCA_gbm <- gbm(positive_revenue ~ .  , data = stockfuture_PCA_train,
                              distribution = "bernoulli", n.trees = 5000,
                              interaction.depth = 1, shrinkage = 0.001,n.minobsinnode = 40,
                              bag.fraction = 0.5, cv.folds = 10)

stockfuture.PCA.gbm.pred.test_raw <- predict.gbm( object = stockfuture_PCA_gbm , newdata = stockfuture_PCA_test)
stockfuture.PCA.gbm.pred.test <- ifelse(stockfuture.PCA.gbm.pred.test_raw >= log(0.6/0.4) , 1,0) 
confusionMatrix(data = factor(stockfuture.PCA.gbm.pred.test) , reference = factor(stockfuture_PCA_test$positive_revenue),positive = "1")
#PCA logit的Accuracy為0.621

pred_PCA_gbm <- prediction(stockfuture.PCA.gbm.pred.test_raw , stockfuture_PCA_test$positive_revenue) 
perf_PCA_gbm <- performance(pred_PCA_gbm , measure = "tpr" , x.measure = "fpr")
performance(pred_PCA_gbm, measure = "auc")@y.values[[1]]
#PCA Gradient Boosting的AUC為0.667

#####LASSO#####
stockfuture %>%
  select(positive_revenue,SP500change,SP500change_lag2,VIXvolume,USDchange,VIXchange,bitchange) -> stockfuture_lasso

stockfuture_lasso_train <- stockfuture_lasso[1:351,]
stockfuture_lasso_test  <- stockfuture_lasso[352:475,]

#logit
stockfuture_lasso_logit <- glm(positive_revenue ~ .  , data = stockfuture_lasso_train , family = binomial("logit"))
summary(stockfuture_lasso_logit)

stockfuture.lasso.logit.pred.test_raw <- predict( stockfuture_lasso_logit , newdata = stockfuture_lasso_test)
stockfuture.lasso.logit.pred.test <- ifelse(stockfuture.lasso.logit.pred.test_raw >= log(0.6/0.4) , 1,0)
confusionMatrix(data = factor(stockfuture.lasso.logit.pred.test) , reference = factor(stockfuture_lasso_test$positive_revenue),positive = "1")
#lasso logit的Accuracy為0.7823

pred_lasso_logit <- prediction(stockfuture.lasso.logit.pred.test_raw , stockfuture_lasso_test$positive_revenue) 
perf_lasso_logit <-performance (pred_lasso_logit , measure = "tpr" , x.measure = "fpr")
performance(pred_lasso_logit, measure = "auc")@y.values[[1]]
#lasso logit的AUC為0.847

#Gradient Boosting
stockfuture_lasso$positive_revenue <- ifelse(stockfuture$nextday_openchange >= 0,1,0)
stockfuture_lasso_train <- stockfuture_lasso[1:351,]
stockfuture_lasso_test  <- stockfuture_lasso[352:475,]

set.seed(1)
stockfuture_lasso_gbm <- gbm(positive_revenue ~ .  , data = stockfuture_lasso_train,
                           distribution = "bernoulli", n.trees = 5000,
                           interaction.depth = 1, shrinkage = 0.001,n.minobsinnode = 40,
                           bag.fraction = 0.5, cv.folds = 10)

stockfuture.lasso.gbm.pred.test_raw <- predict.gbm( object = stockfuture_lasso_gbm , newdata = stockfuture_lasso_test)
stockfuture.lasso.gbm.pred.test <- ifelse(stockfuture.lasso.gbm.pred.test_raw >= log(0.6/0.4) , 1,0) 
confusionMatrix(data = factor(stockfuture.lasso.gbm.pred.test) , reference = factor(stockfuture_lasso_test$positive_revenue),positive = "1")
#lasso logit的Accuracy為0.7581

pred_lasso_gbm <- prediction(stockfuture.lasso.gbm.pred.test_raw , stockfuture_lasso_test$positive_revenue) 
perf_lasso_gbm <- performance(pred_lasso_gbm , measure = "tpr" , x.measure = "fpr")
performance(pred_lasso_gbm, measure = "auc")@y.values[[1]]
#lasso Gradient Boosting的AUC為0.852

#####RF#####
stockfuture %>%
  select(positive_revenue,VIXchange,SP500change,SP500change_lag1,VIXvolume,ETHchange,openchange_lag1) -> stockfuture_RF

stockfuture_RF_train <- stockfuture_RF[1:351,]
stockfuture_RF_test  <- stockfuture_RF[352:475,]

#logit
stockfuture_RF_logit <- glm(positive_revenue ~ .  , data = stockfuture_RF_train , family = binomial("logit"))
summary(stockfuture_RF_logit)

stockfuture.RF.logit.pred.test_raw <- predict( stockfuture_RF_logit , newdata = stockfuture_RF_test)
stockfuture.RF.logit.pred.test <- ifelse(stockfuture.RF.logit.pred.test_raw >= log(0.6/0.4) , 1,0)
confusionMatrix(data = factor(stockfuture.RF.logit.pred.test) , reference = factor(stockfuture_RF_test$positive_revenue),positive = "1")
#RF logit的Accuracy為0.75

pred_RF_logit <- prediction(stockfuture.RF.logit.pred.test_raw , stockfuture_RF_test$positive_revenue) 
perf_RF_logit <-performance (pred_RF_logit , measure = "tpr" , x.measure = "fpr")
performance(pred_RF_logit, measure = "auc")@y.values[[1]]
#RF logit的AUC為0.819

#Gradient Boosting
stockfuture_RF$positive_revenue <- ifelse(stockfuture$nextday_openchange >= 0,1,0)
stockfuture_RF_train <- stockfuture_RF[1:351,]
stockfuture_RF_test  <- stockfuture_RF[352:475,]

set.seed(1)
stockfuture_RF_gbm <- gbm(positive_revenue ~ .  , data = stockfuture_RF_train,
                             distribution = "bernoulli", n.trees = 5000,
                             interaction.depth = 1, shrinkage = 0.001,n.minobsinnode = 40,
                             bag.fraction = 0.5, cv.folds = 10)

stockfuture.RF.gbm.pred.test_raw <- predict.gbm( object = stockfuture_RF_gbm , newdata = stockfuture_RF_test)
stockfuture.RF.gbm.pred.test <- ifelse(stockfuture.RF.gbm.pred.test_raw >= log(0.6/0.4) , 1,0) 
confusionMatrix(data = factor(stockfuture.RF.gbm.pred.test) , reference = factor(stockfuture_RF_test$positive_revenue),positive = "1")
#RF logit的Accuracy為0.782

pred_RF_gbm <- prediction(stockfuture.RF.gbm.pred.test_raw , stockfuture_RF_test$positive_revenue) 
perf_RF_gbm <- performance(pred_RF_gbm , measure = "tpr" , x.measure = "fpr")
performance(pred_RF_gbm, measure = "auc")@y.values[[1]]
#RF Gradient Boosting的AUC為0.856

#####LASSO和RF的交集變數#####

stockfuture %>%
  select(positive_revenue,VIXchange,SP500change,VIXvolume) -> stockfuture_core

stockfuture_core_train <- stockfuture_core[1:351,]
stockfuture_core_test  <- stockfuture_core[352:475,]

#logit
stockfuture_core_logit <- glm(positive_revenue ~ .  , data = stockfuture_core_train , family = binomial("logit"))
summary(stockfuture_core_logit)

stockfuture.core.logit.pred.test_raw <- predict( stockfuture_core_logit , newdata = stockfuture_core_test)
stockfuture.core.logit.pred.test <- ifelse(stockfuture.core.logit.pred.test_raw >= log(0.6/0.4) , 1,0)
confusionMatrix(data = factor(stockfuture.core.logit.pred.test) , reference = factor(stockfuture_core_test$positive_revenue),positive = "1")
#core logit的Accuracy為0.7903

pred_core_logit <- prediction(stockfuture.core.logit.pred.test_raw , stockfuture_core_test$positive_revenue) 
perf_core_logit <-performance (pred_core_logit , measure = "tpr" , x.measure = "fpr")
performance(pred_core_logit, measure = "auc")@y.values[[1]]
#core logit的AUC為0.828

#Gradient Boosting
stockfuture_core$positive_revenue <- ifelse(stockfuture$nextday_openchange >= 0,1,0)
stockfuture_core_train <- stockfuture_core[1:351,]
stockfuture_core_test  <- stockfuture_core[352:475,]

set.seed(1)
stockfuture_core_gbm <- gbm(positive_revenue ~ .  , data = stockfuture_core_train,
                          distribution = "bernoulli", n.trees = 5000,
                          interaction.depth = 1, shrinkage = 0.001,n.minobsinnode = 40,
                          bag.fraction = 0.5, cv.folds = 10)

stockfuture.core.gbm.pred.test_raw <- predict.gbm( object = stockfuture_core_gbm , newdata = stockfuture_core_test)
stockfuture.core.gbm.pred.test <- ifelse(stockfuture.core.gbm.pred.test_raw >= log(0.6/0.4) , 1,0) 
confusionMatrix(data = factor(stockfuture.core.gbm.pred.test) , reference = factor(stockfuture_core_test$positive_revenue),positive = "1")
#core logit的Accuracy為0.774

pred_core_gbm <- prediction(stockfuture.core.gbm.pred.test_raw , stockfuture_core_test$positive_revenue) 
perf_core_gbm <- performance(pred_core_gbm , measure = "tpr" , x.measure = "fpr")
performance(pred_core_gbm, measure = "auc")@y.values[[1]]
#core Gradient Boosting的AUC為0.856


#####結論#####
#就分類問題而言，篩選變數的結果都比較差，不篩選變數直接做Gradient Boosting比較好
