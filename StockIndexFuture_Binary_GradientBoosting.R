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
}

stockfuture <- read_excel("C:/RRR/完整資料.xlsx")
stockfuture$positive_revenue <- ifelse(stockfuture$nextday_openchange >= 0,1,0)

summary(stockfuture)
#從summary來看，58%是正報酬，42%是負報酬，接近1:1

stockfuture_train <- stockfuture[1:330,]
stockfuture_test  <- stockfuture[331:475,]

#####Probit#####
stockfuture_probit <- glm(positive_revenue ~ . - date - nextday_openchange , data = stockfuture_train , family = binomial(probit))
summary(stockfuture_probit)
#目前遇到algorithm did not converge的問題，可能某些變數的線性組合能完美的預測漲跌

coeftest(stockfuture_probit, vcov. = vcovHC, type = "HC3")
logLik(stockfuture_probit)
probitmfx(positive_revenue ~ . - date - nextday_openchange , data = stockfuture_train, atmean = FALSE) #APE instead of PEA

#####Gradient Boosting#####
set.seed(1)
stockfuture_binary_gbm_1 <- gbm(positive_revenue ~ . - date - nextday_openchange , data = stockfuture_train,
                         distribution = "bernoulli", n.trees = 10000,
                         interaction.depth = 1, shrinkage = 0.001,
                         bag.fraction = 0.5, cv.folds = 10)
summary(stockfuture_binary_gbm_1)


stockfuture.binary.gbm.1.pred.train <- ifelse(predict.gbm( object = stockfuture_binary_gbm_1 , newdata = stockfuture_train)>=0,1,0)
stockfuture.binary.gbm.1.pred.test_raw <- predict.gbm( object = stockfuture_binary_gbm_1 , newdata = stockfuture_test)
stockfuture.binary.gbm.1.pred.test <- ifelse(stockfuture.binary.gbm.1.pred.test_raw >=0 , 1,0)

#我們用的是Bernoulli Loss，他的回傳值是Log odds，因此若pred>0則為正，pred<0則為負

confusionMatrix(data = factor(stockfuture.binary.gbm.1.pred.train) , reference = factor(stockfuture_train$positive_revenue))

confusionMatrix(data = factor(stockfuture.binary.gbm.1.pred.test) , reference = factor(stockfuture_test$positive_revenue))
#Test Data的Accuracy為 0.7655 ，Sensitivity為0.5397，Specificity為0.9390
#從實際結果來看，如果實際報酬為正，我們很高機率可以預測正確(0.9390)，如果實際報酬為負，我們只有較低機率預測正確(0.5397)
#從預測的值來看，如果我們預測報酬為負，那預測正確的機率為0.87，如果我們預測報酬為正，那預測正確機率只有0.726
#或著是說TPR為0.9390，FPR為0.

pred <- prediction(stockfuture.binary.gbm.1.pred.test , stockfuture_test$positive_revenue) 
perf <-performance (pred , measure = "tpr" , x.measure = "fpr")
auc <- performance(pred, "auc")


plot(perf , col = "red" , main = "ROC curve" , xlab = "Specificity(FPR)" , ylab = "Sensitivity(TPR)")
abline(0, 1)
text(0.5, 0.8, as.character(round(auc@y.values[[1]],3)))
