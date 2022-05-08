#####Setting#####
{
library(gbm)
library(readxl)
library(MASS)
library(doParallel)
library(DAAG)
library(caret)
}

stockfuture <- read_excel("C:/RRR/完整資料.xlsx")
summary(stockfuture)

stockfuture <- stockfuture[,c(-4:-8,-10:-14,-21,-22,-23,-24,-25)]#如果不放落遲項的話

stockfuture_train <- stockfuture[1:330,]
stockfuture_test  <- stockfuture[331:475,]

rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}

plot(stockfuture$today_openchange)
#####OLS#####
set.seed(1)
stockfuture_OLS <- lm(data = stockfuture_train, nextday_openchange ~ . - date)
summary(stockfuture_OLS)
#R^2為0.4339，OLS樣本內MSE為0.806

stockfuture.OLS.pred.test  <- predict(stockfuture_OLS, newdata = stockfuture_test)
rmse(stockfuture.OLS.pred.test, stockfuture_test$nextday_openchange)
#OLS樣本外MSE為0.56

#同場加映，台指期與美股的相對關係
summary(lm(SP500change ~ . , data = stockfuture_train))
summary(lm(SP500change ~ today_openchange + today_indaychange + today_volume, data = stockfuture_train))
summary(lm(nextday_openchange ~ SP500change , data = stockfuture_train))
#台股受美股影響比較大

#####Gradient Boosting#####

#Tuning
ctrl <- trainControl(method = "repeatedcv",number = 10, repeats = 2, allowParallel = T) #重複的cv，每次分10組，重複2次
registerDoParallel(detectCores()-1)
grid <- expand.grid(n.trees = c(5000,10000,20000), interaction.depth=c(1:3), shrinkage=c(0.05,0.01,0.001) , n.minobsinnode=c(5,10,20))

# set.seed(1)
# stockfuture.gbm.caret <- caret::train(nextday_openchange ~ . - date , data = stockfuture_train, method = "gbm", metric = "RMSE",
#                                       trControl = ctrl, tuneGrid = grid)
# print(stockfuture.gbm.caret)
#從結果來看，n.trees = 10000，shrinkage=0.001，interaction.depth=1，n.minobsinnode=5會使得樣本內的RMSE在CV下最小化，達到0.8361

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

sum(summary(stockfuture_gbm)$rel.inf)#合計為100，代表每個變數影響力所佔的百分比
summary(stockfuture_gbm)
#可以看出各個變數的相對重要性
#在本模型中，SP500和VIX的價格變化幅度是最重要的變數，遠超其他變數
#但要注意這個重要性可能會受到shrinkage和n.trees的數值所影響

stockfuture.gbm.pred.train <- predict(stockfuture_gbm, newdata = stockfuture_train)
rmse(stockfuture.gbm.pred.train, stockfuture_train$nextday_openchange)
#Gradient Boosting樣本內MSE為0.884

stockfuture.gbm.pred.test  <- predict(stockfuture_gbm, newdata = stockfuture_test)
rmse(stockfuture.gbm.pred.test, stockfuture_test$nextday_openchange)  
#Gradient Boosting樣本外MSE為0.482

#從結果來看，不放落遲項的預測能力比較好，而且放了會有overfitting的問題

