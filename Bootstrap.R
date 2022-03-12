{
  library(tidyr)
  library(caret)
  library(car)
  library(lmboot)
}

setwd("D:/RRR")
Auto <- read.csv("Auto.csv")

Auto = Auto[Auto$horsepower!="?",]
Auto$horsepower <- as.numeric(Auto$horsepower)
totalnum <- nrow(Auto)

#####Q1(a)#####
vsaidx <- sample(c(1:totalnum),totalnum/2)
vsatrain <- Auto[vsaidx,]
vsatest <- Auto[-vsaidx,]

vsafit_1 <- lm(mpg~horsepower , data = vsatrain)
vsafit_2 <- lm(mpg~horsepower+weight , data = vsatrain)
vsafit_3 <- lm(mpg~horsepower+weight+acceleration , data = vsatrain)

vsamse.1.i <- mean((vsatrain$mpg-predict(vsafit_1,vsatrain))^2)
vsamse.2.i <- mean((vsatrain$mpg-predict(vsafit_2,vsatrain))^2)
vsamse.3.i <- mean((vsatrain$mpg-predict(vsafit_3,vsatrain))^2)
c(vsamse.1.i,vsamse.2.i,vsamse.3.i)

vsamse.1.o <- mean((vsatest$mpg-predict(vsafit_1,vsatest))^2)
vsamse.2.o <- mean((vsatest$mpg-predict(vsafit_2,vsatest))^2)
vsamse.3.o <- mean((vsatest$mpg-predict(vsafit_3,vsatest))^2)
c(vsamse.1.o,vsamse.2.o,vsamse.3.o)
#第三個模型的樣本外預測MSE最小

#####Q1(b)#####
ctrl <- trainControl(method = "LOOCV")
loocvfit_1 <- train (mpg~horsepower, data = Auto, method = "lm", trControl = ctrl)
loocvfit_2 <- train (mpg~horsepower+weight, data = Auto, method = "lm", trControl = ctrl)
loocvfit_3 <- train (mpg~horsepower+weight+acceleration, data = Auto, method = "lm", trControl = ctrl)

c(loocvfit_1$results$RMSE^2,loocvfit_2$results$RMSE^2,loocvfit_3$results$RMSE^2)
#第三個模型的樣本外預測MSE最小

#####Q1(c)#####
set.seed(100)
ctrl <- trainControl(method = "cv" , number = 10)
loocvfit_1 <- train (mpg~horsepower, data = Auto, method = "lm", trControl = ctrl)
loocvfit_2 <- train (mpg~horsepower+weight, data = Auto, method = "lm", trControl = ctrl)
loocvfit_3 <- train (mpg~horsepower+weight+acceleration, data = Auto, method = "lm", trControl = ctrl)

c(loocvfit_1$results$RMSE^2,loocvfit_2$results$RMSE^2,loocvfit_3$results$RMSE^2)
#第三個模型的樣本外預測MSE最小

#####Q2(a)#####
reg <- lm(mpg ~ horsepower , data = Auto )
summary(reg)
reg$coefficients[2]
#0.1191

bpair_coef = c()
for (i in 1:1000){
  pairedidx = sample(1:totalnum,totalnum,replace = T)
  reg_pair <- lm( mpg ~ horsepower , data = Auto[pairedidx,])
  bpair_coef[i] <-reg_pair$coefficients[2]
}
c(mean(bpair_coef),
  sqrt(sum((bpair_coef-mean(bpair_coef))^2)/(1000-1))
)
#0.1193 0.0090

#####Q2(b)#####
bres_coef = c()
for (i in 1:1000){
  residx = sample(1:totalnum,totalnum,replace = T)
  bresdata = data.frame(mpg = reg$fitted.values + reg$residuals[residx] , horsepower = Auto$horsepower)
  reg_res <- lm( mpg ~ horsepower,data = bresdata)
  bres_coef[i] <-reg_res$coefficients[2]
}
c(mean(bres_coef),sqrt(sum((bres_coef-mean(bres_coef))^2)/(999)))
#0.1190 0.01174