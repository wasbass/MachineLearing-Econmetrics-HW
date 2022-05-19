#####Package#####
{
library(gbm)
library(readxl)
library(MASS)
library(doParallel)
library(DAAG)
library(caret)
library(dplyr)
library(tseries)
}

#####Setting#####
setwd("C:/RRR/")
stockfuture <- read_excel("完整資料.xlsx")
summary(stockfuture)

stockfuture_train <- stockfuture[1:351,]
stockfuture_test  <- stockfuture[352:475,]

rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}

plot(stockfuture$trend,stockfuture$today_openchange)

#####Without feature selecting#####

#Tuning
ctrl <- trainControl(method = "repeatedcv",number = 10, repeats = 2, allowParallel = T) #重複的cv，每次分10組，重複2次
registerDoParallel(detectCores()-1)
grid <- expand.grid(n.trees = c(5000,10000,20000), interaction.depth=c(1:3), shrinkage=c(0.05,0.01,0.001) , n.minobsinnode=c(10,25,50))

# set.seed(1)
# stockfuture.gbm.caret <- caret::train(nextday_openchange ~ . - date , data = stockfuture_train, method = "gbm", metric = "RMSE",
#                                       trControl = ctrl, tuneGrid = grid)
# print(stockfuture.gbm.caret)
#從結果來看，n.trees = 10000，shrinkage=0.001，interaction.depth=1，n.minobsinnode=50會使得樣本內的RMSE在CV下最小化，達到0.8361

#Build Model
set.seed(1)
stockfuture_gbm <- gbm(nextday_openchange ~ . - date , data = stockfuture_train,
                         distribution = "gaussian", n.trees = 10000,
                         interaction.depth = 1, shrinkage = 0.001,
                         bag.fraction = 0.5, cv.folds = 10,n.minobsinnode=50)
#在Tuning，也就是優化參數的過程中，由training的RMSE說明n.minobsinnode越小越好
#但從樣本外MSE來看，n.minobsinnode過小會存在overfitting的問題
#在我們持續加大n.minobsinnode後，樣本外MSE反而越來越小
#因此我們最後選擇用小的shrinkage，大的n.trees來增強學習能力
#並且用小的interaction.depth，大的n.minobsinnode來避免過度配適問題
#之後的control都只會考慮n.tree=c(2500,5000,10000)，interaction.depth=c(1,2)，並且把n.minobsinnode固定在50，shrinkage固定在0.001

sum(summary(stockfuture_gbm)$rel.inf)#合計為100，代表每個變數影響力所佔的百分比
summary(stockfuture_gbm)
#可以看出各個變數的相對重要性
#在本模型中，SP500和VIX的價格變化幅度是最重要的變數，遠超其他變數
#但要注意這個重要性可能會受到shrinkage和n.trees的數值所影響

stockfuture.gbm.pred.train <- predict(stockfuture_gbm, newdata = stockfuture_train)
rmse(stockfuture.gbm.pred.train, stockfuture_train$nextday_openchange)
#Gradient Boosting樣本內MSE為0.849

stockfuture.gbm.pred.test  <- predict(stockfuture_gbm, newdata = stockfuture_test)
rmse(stockfuture.gbm.pred.test, stockfuture_test$nextday_openchange)  
#Gradient Boosting樣本外MSE為0.492

#從結果來看，不放落遲項的預測能力比較好，而且放了會有overfitting的問題


#####PCA#####
PCA <- read.csv("PCA.csv")
stockfuture_PCA <- cbind( nextday_openchange =  stockfuture$nextday_openchange , PCA)

stockfuture_PCA_train <- stockfuture_PCA[1:351,]
stockfuture_PCA_test  <- stockfuture_PCA[352:475,]

#Tuning
ctrl <- trainControl(method = "repeatedcv",number = 10, repeats = 2, allowParallel = T) #重複的cv，每次分10組，重複2次
registerDoParallel(detectCores()-1)
grid <- expand.grid(n.trees = c(2500,5000,10000), interaction.depth=c(1:2), shrinkage=c(0.001) , n.minobsinnode=c(50))

#set.seed(1)
#stockfuture.gbm.PCA.caret <- caret::train(nextday_openchange ~ . , data = stockfuture_PCA_train, method = "gbm", metric = "RMSE",
#                                       trControl = ctrl, tuneGrid = grid)
#print(stockfuture.gbm.PCA.caret)
#篩選變數後，n.tree=5000，interaction.depth=2

set.seed(1)
stockfuture_PCA_gbm <- gbm(nextday_openchange ~ . , data = stockfuture_PCA_train,
                       distribution = "gaussian", n.trees = 5000,
                       interaction.depth = 2, shrinkage = 0.001,
                       bag.fraction = 0.5, cv.folds = 10,n.minobsinnode=50)

istockfuture.PCA.gbm.pred.train  <- predict(stockfuture_PCA_gbm, newdata = stockfuture_PCA_train)
rmse(stockfuture.PCA.gbm.pred.train, stockfuture_PCA_train$nextday_openchange)
#樣本內MSE為0.886

stockfuture.PCA.gbm.pred.test  <- predict(stockfuture_PCA_gbm, newdata = stockfuture_PCA_test)
rmse(stockfuture.PCA.gbm.pred.test, stockfuture_PCA_test$nextday_openchange)  
#PCA的Gradient Boosting樣本外MSE為0.598

#####LASSO#####
stockfuture %>%
  dplyr::select(nextday_openchange,openchange_lag1,SP500change,SP500change_lag4,USDchange,VIXchange,bitchange) -> stockfuture_LASSO

stockfuture_LASSO_train <- stockfuture_LASSO[1:351,]
stockfuture_LASSO_test  <- stockfuture_LASSO[352:475,]

#Tuning
ctrl <- trainControl(method = "repeatedcv",number = 10, repeats = 2, allowParallel = T) #重複的cv，每次分10組，重複2次
registerDoParallel(detectCores()-1)
grid <- expand.grid(n.trees = c(2500,5000,10000), interaction.depth=c(1:2), shrinkage=c(0.001) , n.minobsinnode=c(50))

#set.seed(1)
#stockfuture.gbm.LASSO.caret <- caret::train(nextday_openchange ~ . , data = stockfuture_LASSO_train, method = "gbm", metric = "RMSE",
#                                          trControl = ctrl, tuneGrid = grid)
#print(stockfuture.gbm.LASSO.caret)
#篩選變數後，n.tree=5000，interaction.depth=1

set.seed(1)
stockfuture_LASSO_gbm <- gbm(nextday_openchange ~ . , data = stockfuture_LASSO_train,
                           distribution = "gaussian", n.trees = 5000,
                           interaction.depth = 1, shrinkage = 0.001,
                           bag.fraction = 0.5, cv.folds = 10,n.minobsinnode=50)y

stockfuture.LASSO.gbm.pred.train  <- predict(stockfuture_LASSO_gbm, newdata = stockfuture_LASSO_train)
rmse(stockfuture.LASSO.gbm.pred.train, stockfuture_LASSO_train$nextday_openchange)  
#樣本內MSE為0.861

stockfuture.LASSO.gbm.pred.test  <- predict(stockfuture_LASSO_gbm, newdata = stockfuture_LASSO_test)
rmse(stockfuture.LASSO.gbm.pred.test, stockfuture_LASSO_test$nextday_openchange)  
#LASSO的Gradient Boosting樣本外MSE為0.492

#####RF#####
stockfuture %>%
  dplyr::select(nextday_openchange,VIXchange,SP500change,SP500volume,today_volume,indaychange_lag2,VIXvolume) -> stockfuture_RF

stockfuture_RF_train <- stockfuture_RF[1:351,]
stockfuture_RF_test  <- stockfuture_RF[352:475,]

#Tuning
ctrl <- trainControl(method = "repeatedcv",number = 10, repeats = 2, allowParallel = T) #重複的cv，每次分10組，重複2次
registerDoParallel(detectCores()-1)
grid <- expand.grid(n.trees = c(2500,5000,10000), interaction.depth=c(1:2), shrinkage=c(0.001) , n.minobsinnode=c(50))

#set.seed(1)
#stockfuture.gbm.RF.caret <- caret::train(nextday_openchange ~ . , data = stockfuture_RF_train, method = "gbm", metric = "RMSE",
#                                            trControl = ctrl, tuneGrid = grid)
#print(stockfuture.gbm.RF.caret)
#n.tree=2500，interaction.depth=2

set.seed(1)
stockfuture_RF_gbm <- gbm(nextday_openchange ~ . , data = stockfuture_RF_train,
                             distribution = "gaussian", n.trees = 2500,
                             interaction.depth = 2, shrinkage = 0.001,
                             bag.fraction = 0.5, cv.folds = 10,n.minobsinnode=50)

stockfuture.RF.gbm.pred.test  <- predict(stockfuture_RF_gbm, newdata = stockfuture_RF_test)
rmse(stockfuture.RF.gbm.pred.test, stockfuture_RF_test$nextday_openchange)  
#RF的Gradient Boosting樣本外MSE為0.474，比原本的好

#加碼再看看樣本內的MSE
stockfuture.RF.gbm.pred.train  <- predict(stockfuture_RF_gbm, newdata = stockfuture_RF_train)
rmse(stockfuture.RF.gbm.pred.train, stockfuture_RF_train$nextday_openchange)  
#RF的Gradient Boosting樣本內MSE為0.881，比原本的差
#代表我們在篩選特徵後，稍微解決了Gradient Boosting原本有的Overfitting問題

#####LASSO和RF的交集變數#####
#只有SP500change和VIXchange
stockfuture %>%
  dplyr::select(nextday_openchange,SP500change,VIXchange) -> stockfuture_core

stockfuture_core_train <- stockfuture_core[1:351,]
stockfuture_core_test  <- stockfuture_core[352:475,]

#Tuning
ctrl <- trainControl(method = "repeatedcv",number = 10, repeats = 2, allowParallel = T) #重複的cv，每次分10組，重複2次
registerDoParallel(detectCores()-1)
grid <- expand.grid(n.trees = c(2500,5000,10000), interaction.depth=c(1:2), shrinkage=c(0.001) , n.minobsinnode=c(50))

#set.seed(1)
#stockfuture.gbm.core.caret <- caret::train(nextday_openchange ~ . , data = stockfuture_core_train, method = "gbm", metric = "RMSE",
#                                         trControl = ctrl, tuneGrid = grid)
#print(stockfuture.gbm.core.caret)
#n.tree=2500，interaction.depth=2

set.seed(1)
stockfuture_core_gbm <- gbm(nextday_openchange ~ . , data = stockfuture_core_train,
                          distribution = "gaussian", n.trees = 2500,
                          interaction.depth = 1, shrinkage = 0.001,
                          bag.fraction = 0.5, cv.folds = 10,n.minobsinnode=50)

stockfuture.core.gbm.pred.train  <- predict(stockfuture_core_gbm, newdata = stockfuture_core_train)
rmse(stockfuture.core.gbm.pred.train, stockfuture_core_train$nextday_openchange) 
#樣本內MSE為0.894

stockfuture.core.gbm.pred.test  <- predict(stockfuture_core_gbm, newdata = stockfuture_core_test)
rmse(stockfuture.core.gbm.pred.test, stockfuture_core_test$nextday_openchange)  
#core的Gradient Boosting樣本外MSE為0.487，比原本的好


#####結論#####
#不做篩選的Gradient Boosting的樣本外預測已經蠻不錯的了
#但是經過Random Forest篩選變數後再做Gradient Boosting，樣本外預測的結果會更好
#代表如果在Gradient Boosting中放入太多不相關的變數，可能會有overfitting的問題
#相較之下，PCA和LASSO的篩選變數結果就不那麼優秀
#而若是拿LASSO和Random Forest都有選到的變數做預測，結果也不錯

#####預測Y_hat#####

newdata_0712 <- read_excel("newdata.xlsx",sheet = "2107-12")
newdata_0104 <- read_excel("newdata.xlsx",sheet = "2201-04")
stockfuture.RF.gbm.pred.0712  <- predict(stockfuture_RF_gbm, newdata = newdata_0712)
rmse(stockfuture.RF.gbm.pred.0712,newdata_0712$nextday_openchange)

stockfuture.RF.gbm.pred.0104  <- predict(stockfuture_RF_gbm, newdata = newdata_0104)
rmse(stockfuture.RF.gbm.pred.0104[1:71],as.numeric(newdata_0104$nextday_openchange[1:71]))

append(stockfuture.RF.gbm.pred.0712 ,stockfuture.RF.gbm.pred.0104)
write.csv(append(stockfuture.RF.gbm.pred.0712 ,stockfuture.RF.gbm.pred.0104),
          file = "predicted_openchange.csv")
