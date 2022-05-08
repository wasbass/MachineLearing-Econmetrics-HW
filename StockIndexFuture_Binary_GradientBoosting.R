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
}

stockfuture <- read_excel("C:/RRR/完整資料.xlsx")

stockfuture <- stockfuture[,c(-4:-8,-10:-14,-21,-22,-23,-24,-25)]#如果不放落遲項的話

###如果設成factor跑gbm，R會當掉
stockfuture$positive_revenue <- as.factor(ifelse(stockfuture$nextday_openchange >= 0,"1","0"))

summary(stockfuture)
#從summary來看，58%是正報酬，42%是負報酬，接近1:1

stockfuture_train <- stockfuture[1:330,]
stockfuture_test  <- stockfuture[331:475,]

#####logit#####

stockfuture_probit <- glm(positive_revenue ~ . - date - nextday_openchange, #- VIXchange - SP500change,
                          data = stockfuture_train , family = binomial(probit)
                          )
summary(stockfuture_probit)
glm(positive_revenue ~ VIXchange + SP500change + VIXvolume + bitchange, data = stockfuture_train , family = binomial(probit))
iv.mult(data.frame(stockfuture_train[,3:21]) , y = "positive_revenue" , summary =TRUE , verbose = FALSE)
#iv.mult(data.frame(stockfuture_train[,3:36]) , y = "positive_revenue" , summary =TRUE , verbose = FALSE)
#probit目前遇到algorithm did not converge的問題，可能某些變數的線性組合能完美的預測漲跌
#可以看到SP500和VIX的變動可能可以完全的預測台指期走勢方向
#這邊改用logit試試看
stockfuture_logit <- glm(positive_revenue ~ . - date - nextday_openchange , data = stockfuture_train , family = binomial("logit"))
summary(stockfuture_logit)
coeftest(stockfuture_logit, vcov. = vcovHC, type = "HC3")
logLik(stockfuture_logit)
logitmfx(positive_revenue ~ . - date - nextday_openchange , data = stockfuture_train, atmean = FALSE) #APE instead of PEA

#觀察預測結果
stockfuture.binary.logit.pred.train_raw <- predict( stockfuture_logit , newdata = stockfuture_train)
stockfuture.binary.logit.pred.train <- ifelse(stockfuture.binary.logit.pred.train_raw >= log(0.6/0.4) , 1,0)

stockfuture.binary.logit.pred.test_raw <- predict( stockfuture_logit , newdata = stockfuture_test)
stockfuture.binary.logit.pred.test <- ifelse(stockfuture.binary.logit.pred.test_raw >= log(0.6/0.4) , 1,0)

confusionMatrix(data = factor(stockfuture.binary.logit.pred.train) , reference = factor(stockfuture_train$positive_revenue),positive = "1")
confusionMatrix(data = factor(stockfuture.binary.logit.pred.test) , reference = factor(stockfuture_test$positive_revenue),positive = "1")
#Accuracy為0.7931
#TPR(true positive rate)為0.8902，TNR(true negative rate)為0.6667
#PPV(positive predictive value)為0.7766，NPV(negative predictive value)為0.8235


#####Gradient Boosting#####

#####Tuning#####
ctrl <- trainControl(method = "repeatedcv",number = 10, repeats = 2, allowParallel = T , verboseIter = TRUE) #重複的cv，每次分10組，重複2次
registerDoParallel(detectCores()-1)
grid <- expand.grid(n.trees = c(2500,5000), interaction.depth=c(1:2), shrinkage=c(0.01,0.005,0.001) , n.minobsinnode=c(40,50,60))

#grid <- expand.grid(n.trees = c(5000,10000), interaction.depth=c(1), shrinkage=c(0.001) , n.minobsinnode=c(5,10,20))
set.seed(1)
stockfuture.gbm.caret <- caret::train(positive_revenue ~ . - date - nextday_openchange, data = stockfuture_train, method = "gbm", metric = "Accuracy",
                                      trControl = ctrl, tuneGrid = grid)
print(stockfuture.gbm.caret)
#n.tree1000時表現不好，超過5000之後準確度下滑，可以考慮interaction.depth為2或1
#n.minobsinnode稍微大點表現越好
#最後決定n.minobsinnode=40，interaction.depth=1(or 2)，shrinkage=0.001，n.tree=5000

#build model#
stockfuture$positive_revenue <- ifelse(stockfuture$nextday_openchange >= 0,1,0)
stockfuture_train <- stockfuture[1:330,]
stockfuture_test  <- stockfuture[331:475,]

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
#Test Data的Accuracy為 0.8069 ，Sensitivity為0.8780，Specificity為0.7143
#從實際結果來看，如果實際報酬為正，我們很高機率可以預測正確(0.8780)，如果實際報酬為負，我們只有較低機率預測正確(0.7143)
#從預測的值來看，如果我們預測報酬為正，那預測正確機率為0.8000，如果我們預測報酬為負，那預測正確的機率為0.8182，
#或著是說TPR(true positive rate)為0.8780，TNR(true negative rate)為0.7143
#以及PPV(positive predictive value)為0.8000，NPV(negative predictive value)為0.8182


#畫ROC圖
pred_logit <- prediction(stockfuture.binary.logit.pred.test_raw , stockfuture_test$positive_revenue) 
perf_logit <-performance (pred_logit , measure = "tpr" , x.measure = "fpr")
performance(pred_logit, measure = "auc")@y.values[[1]]

pred_GB <- prediction(stockfuture.binary.gbm.pred.test_raw , stockfuture_test$positive_revenue) 
perf_GB <-performance (pred_GB , measure = "tpr" , x.measure = "fpr")
performance(pred_GB, measure = "auc")@y.values[[1]]

windows()
plot(perf_GB , col = "red" , main = "ROC curve" , xlab = "1-Specificity (FPR)" , ylab = "Sensitivity(TPR)" , lwd = 5)
plot(perf_logit , col = "blue" , lwd = 5 , add = TRUE)
abline(0, 1 , lwd = 2)
legend("bottomright",legend = c("GradientBoosting(AUC = 0.863)","Logistic(AUC = 0.827)"), lty = 1 , col = c("red","blue") , lwd = 5)
#Gradient Boosting的AUC為0.864，還算不錯，比logistic高上一些

