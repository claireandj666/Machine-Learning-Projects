cat("\014")

#install.packages("lattice")
library(class)
library(ggplot2)
library(dplyr)
library(glmnet)
library(MASS)
library(caret)
library(lattice)

install.packages("rmutil")
install.packages("tictoc")
install.packages("latex2exp")

library(rmutil)
library(tictoc)
library(latex2exp)
library(e1071)


ozone = read.csv("/Users/yandichen/Downloads/eighthr.data", header=FALSE)
head(ozone)
summary(ozone)


# remove "?" values 
newozone = ozone %>% 
  na_if("?") %>%
  na.omit

newozone1 = newozone %>% 
  mutate_all(~ifelse(. %in% c("N/A", "null", ""), NA, .)) %>% 
  na.omit()

summary(newozone1)
head(newozone1)
#na.omit(mutate_all(newozone, ~ifelse(. %in% c("N/A", "null", ""),  NA, .)))

X = model.matrix(V74~., newozone1)[, -1]
y =    factor(newozone$V74)
n =    dim(X)[1] # sample size
p =    dim(X)[2] # number of predictors/features
S =   50 

set.seed(1)

X = scale(X)

# imbalance
sum(y==0)     # 1719
sum(y==1)     # 128
sum(y==0)/n   # 0.9306984
sum(y==1)/n   # 0.06930157

# Err is S x 6 matrix
# column 1 of Err = total train error
# column 2 of Err = total test error

Err.lasso     = matrix(0, nrow = S, ncol = 2) 
Err.ridge     = matrix(0, nrow = S, ncol = 2) 
Err.rf        = matrix(0, nrow = S, ncol = 2) 
Err.logi     = matrix(0, nrow = S, ncol = 2) 
Err.svm      = matrix(0, nrow = S, ncol = 2) 

Err.cv.lasso     = matrix(0, nrow = S, ncol =1 ) 
Err.cv.ridge     = matrix(0, nrow = S, ncol =1) 
Err.cv.svm       = matrix(0, nrow = S, ncol =1) 

# last step, avg(time.lasso())
time.lasso = matrix(0, nrow = S)
time.ridge = matrix(0, nrow = S) 
time.rf    = matrix(0, nrow = S) 
time.logi  = matrix(0, nrow = S) 
time.svm   = matrix(0, nrow = S) 

#mean(time.lasso)
#mean(time.ridge)
#mean(time.rf)
#mean(time.logi)
#mean(time.svm)

for (s in 1:S) {
  
  # randomly splitting the data into test and train set
  random_order  =  sample(n)
  n.train       =  floor(n*0.5)
  n.test        =  n-n.train
  trainSet      =  random_order[1:n.train]
  testSet       =  random_order[(1+n.train):n]
  y.test        =  y[testSet]
  X.test        =  X[testSet, ]
  y.train       =  y[trainSet]
  X.train       =  X[trainSet, ]
  y.os.train    =  y.train   # initialize the over-sampled (os) set to train the models
  X.os.train    =  X.train   # initialize the over-sampled (os) set to train the models
  
  
  
  # to take into account the imbalance
  # below we over-sample (with replacement) the  data so that the data is balanced
  imbalance     =     FALSE   
  if (imbalance == TRUE) {
    index.yis0      =      which(y.train==0)  # idetify the index of  points with label 0
    index.yis1      =      which(y.train==1) # idetify the index of  points with label 1
    n.train.1       =      length(index.yis1)
    n.train.0       =      length(index.yis0)
    if (n.train.1 > n.train.0) {     # we need more 0s in our training set, so we over sample with replacement
      more.train    =      sample(index.yis0, size=n.train.1-n.train.0, replace=TRUE)
    }         
    else {    # we need more 1s in our training set, so we over sample with replacement          
      more.train    =      sample(index.yis1, size=n.train.0-n.train.1, replace=TRUE)
    }
    
    
    ##### the code below CORRECTLY over samples the train set 
    ##### and stores it in y.train_ and X.train_
    y.os.train        =       as.factor(c(y.train, y.train[more.train])-1) 
    X.os.train        =       rbind2(X.train, X.train[more.train,]) 
    
  }
  
  
  
  ###################################  1. logistic regression ################################## ################################## 
  st = proc.time()
  
  os.train.data = data.frame(X.os.train, as.factor(y.os.train))
  names(os.train.data)[74]= "y"
  
  X.train1 = data.frame(X.train)
  X.test1 = data.frame(X.test)
  logi.fit = glm(y~., data = os.train.data, family = binomial)
  logi.prob = predict(logi.fit, X.train1, response='response')
  y.train.hat = rep(0, length(y.train))
  y.train.hat[logi.prob>0.5]=1
  table(y.train, y.train.hat)
  logi.prob2 = predict(logi.fit, X.test1, response='response')
  y.test.hat = rep(0, length(y.test))
  y.test.hat[logi.prob2>0.5]=1
  
  Err.logi[s,1] = mean(y.train != y.train.hat)
  Err.logi[s,2] = mean(y.test != y.test.hat)
  
  time.period = proc.time() - st
  time.logi[s,1] = time.period["elapsed"]
  
  #  logis = glm(V74~., X.train, family = binomial)
  
  
  
  ##################################  2. optimize Lasso logistic regression using cross validation ################################## 
  st = proc.time()
  
  m                 =     10
  lasso.cv          =     cv.glmnet(X.os.train, y.os.train, family = "binomial", alpha = 1,  nfolds = 10, type.measure="class")
  lasso.fit         =     glmnet(X.os.train, y.os.train, lambda = lasso.cv$lambda.min, family = "binomial", alpha = 1)
  
  y.train.hat       =     predict(lasso.fit, newx = X.train, type = "class")
  y.test.hat        =     predict(lasso.fit, newx = X.test, type = "class")
  Err.lasso[s,1]    =     mean(y.train != y.train.hat)
  Err.lasso[s,2]    =     mean(y.test != y.test.hat)
  
  #plot(lasso.cv)
  
  # min cv err
  Err.cv.lasso[s,1] = min(lasso.cv$cvm)
  #new_row = list(Model = "lasso", RunNumber = S,ErrorRate = Err.cv.lasso )
  #err.cv = rbind(err.cv, new_row, stringsAsFactors = FALSE)
  
  
  time.period = proc.time() - st
  time.lasso[s,1] = time.period["elapsed"]
  
  
  
  
  
  
  ###################################  3. optimize ridge logistic regression using cross validation ################################## 
  st = proc.time()
  
  m                 =     10
  ridge.cv          =     cv.glmnet(X.os.train, y.os.train, family = "binomial", alpha = 0,  nfolds = m, type.measure="class")
  ridge.fit         =     glmnet(X.os.train, y.os.train, lambda = ridge.cv$lambda.min, family = "binomial", alpha = 0)
  y.train.hat       =     predict(ridge.fit, newx = X.train, type = "class")
  y.test.hat        =     predict(ridge.fit, newx = X.test, type = "class")
  Err.ridge[s,1]    =     mean(y.train != y.train.hat)
  Err.ridge[s,2]    =     mean(y.test != y.test.hat)
  
  time.period = proc.time() - st
  time.ridge[s,1] = time.period["elapsed"]
  
  # min cv err
  Err.cv.ridge[s,1] = min(ridge.cv$cvm)
  #new_row = list(Model = "ridge", RunNumber = S,ErrorRate = Err.cv.ridge )
  #err.cv = rbind(err.cv, new_row, stringsAsFactors = FALSE)
  
  #Err.cv.ridge[s,1] = ridge.cv$cvm
  
  
  
  ###################################  4. random forrest ################################## ################################## ################################## 
  # alternative way of breaking data into train and test 
  st = proc.time()
  
  os.train.data2           =      data.frame(X.os.train, as.factor(y.os.train))
  train.data              =      data.frame(X.train, as.factor(y.train))
  test.data               =      data.frame(X.test, as.factor(y.test))
  names(os.train.data2)[74]=      "y"
  names(train.data)[74]   =      "y"
  names(test.data)[74]    =      "y"
  library(randomForest)
  rf.fit            =     randomForest(y~., data = os.train.data2, mtry = sqrt(p), ntree=300)
  y.train.hat       =     predict(rf.fit, newdata = train.data)
  y.test.hat        =     predict(rf.fit, newdata = test.data)
  Err.rf[s,1]       =     mean(y.train != y.train.hat)
  Err.rf[s,2]       =     mean(y.test != y.test.hat)
  
  time.period = proc.time() - st
  time.rf[s,1] = time.period["elapsed"]
  
  
  
  
  
  
  
  ###################################  5. svm ################################## ################################## ################################## 
  st = proc.time()
  os.train.data1 = data.frame(X.os.train, as.factor(y.os.train))
  names(os.train.data1)[74]= "y"
  
  #train.data              =      data.frame(X[trainSet, ], as.factor(y[trainSet]))
  #test.data               =      data.frame(X[testSet, ], as.factor(y[testSet]))
  
  tune.svm = tune(svm, y~., data = os.train.data1, kernel="radial",
                  range = list(cost = 10^seq(-2, 2, length.out = 5),
                               gamma = 10^seq(-2,2,length.out = 5)))
  summary(tune.svm)
  
  svm.fit = tune.svm$best.model
  
  y.train.hat = predict(svm.fit, X.train)
  y.test.hat = predict(svm.fit, X.test)
  
  table(y.train, y.train.hat)
  table(y.test, y.test.hat)
  
  Err.svm[s,1]       =     mean(y.train != y.train.hat)
  Err.svm[s,2]       =     mean(y.test != y.test.hat)
  
  time.period = proc.time() - st
  time.svm[s,1] = time.period["elapsed"]
  
  # min cv err
  #Err.cv.svm = min(tune.svm$best.performance)
  Err.cv.svm[s,1] = tune.svm$best.performance
  #new_row = list(Model = "SVM", RunNumber = S,ErrorRate = Err.cv.svm )
  #err.cv = rbind(err.cv, new_row, stringsAsFactors = FALSE)
  
  
  cat(sprintf("s=%1.f| Test:rf=%.2f| l1=%.2f,l2=%.2f|  ||| Train:rf=%.2f| l1=%.2f,l2=%.2f\n", 
              s, Err.rf[s,2], Err.lasso[s,2],Err.ridge[s,2], Err.rf[s,1], Err.lasso[s,1],Err.ridge[s,1]   ))
  
}


# Err is S x 6 matrix
# column 1 of Err = train error
# column 2 of Err = test error
################################## ################################## ################################## ################################## ##################
err.train           =     data.frame(c(rep("logistic", S), rep("rf", S),  rep("logistic lasso", S),  rep("logistic ridge", S), rep("svm", S)) , 
                                     c(Err.logi[, 1], Err.rf[, 1], Err.lasso[, 1], Err.ridge[, 1], Err.svm[, 1]))
err.test            =     data.frame(c(rep("logistic", S), rep("rf", S),  rep("logistic lasso", S),  rep("logistic ridge", S), rep("svm", S)) , 
                                     c(Err.logi[, 2], Err.rf[, 2], Err.lasso[, 2], Err.ridge[, 2], Err.svm[, 2] ))

err.cv        =     data.frame(c(rep("logistic lasso", S),  rep("logistic ridge", S), rep("svm", S)) , 
                               c(Err.cv.lasso[, 1], Err.cv.ridge[, 1], Err.cv.svm[, 1] ))

#err.cv        =     data.frame(Model = factor(), RunNumber = integer(),
#                               ErrorRate = double(),
#                               stringsAsFactors = FALSE)

#time.total = data.frame()




colnames(err.train)    =     c("method","err")
colnames(err.test)     =     c("method","err")
colnames(err.cv)     =     c("method","err")

p1 = ggplot(err.train)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()  +
  ggtitle("train errors, 0.5n") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+ ylim(0, 0.15)  

p2 = ggplot(err.test)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()   +
  ggtitle("test errors, 0.5n") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+ ylim(0, 0.15)  

p3 = ggplot(err.cv)   +     aes(x=method, y = err, fill=method) +   geom_boxplot()   +
  ggtitle("minimum cv error, 0.5n") +
  theme( axis.title.x = element_text(size = 12, face     = "bold", family = "Courier"),
         plot.title          = element_text(size = 12, family   = "Courier"), 
         axis.title.y        = element_text(size = 12, face     = "bold", family = "Courier"), 
         axis.text.x         = element_text(angle= 45, hjust    = 1, size = 10, face = "bold", family = "Courier"), 
         axis.text.y         = element_text(angle= 45, vjust    = 0.7, size = 10, face = "bold", family = "Courier"))+ ylim(0, 0.15)  



Err.logi
Err.lasso
Err.ridge
Err.rf
Err.svm


library("gridExtra")
grid.arrange(p1, p2,p3, ncol=4)

svm_cv.plot = ggplot(data = tune.svm$performances, aes(x=cost, y=gamma)) + geom_tile(aes(fill=error)) + 
  scale_x_continuous(trans = 'log10') + scale_y_continuous(trans = 'log10') + ggtitle("SVM CV Heatmap, 0.5n") + 
  scale_fill_distiller(palette = "Spectral")




################################## ################################## ##################################  before DO: coz mess up with barplot
# coefficient
lasso.coef = as.matrix(lasso.fit$beta)
ridge.coef = as.matrix(ridge.fit$beta)

par(mfrow=c(1,2))
barplot(lasso.coef, beside = TRUE, horiz = TRUE, col = 'lightblue',axes = TRUE,
        xlab= 'Lasso coefficient, 0.5n', names.arg = 'Lasso Attributes: V1 - V73')
barplot(ridge.coef, beside = TRUE, horiz = TRUE, col = 'mistyrose',axes = TRUE,
        xlab = 'Ridge coefficient, 0.5n', names.arg = 'Ridge Attributes: V1 - V73')

lasso.coef == max(lasso.coef)
sort(lasso.coef)





# 10-fold cv curve: lasso, ridge, svm

lasso.fit.0          =     glmnet(X.os.train, y.os.train, lambda =0, family = "binomial", alpha = 1)
n.lambdas = dim(lasso.fit$beta)[2]
lasso.beta.ratio     = rep(0, n.lambdas)

for (i in 1:n.lambdas) {
  lasso.beta.ratio[i]   =   sum(abs(lasso.fit$beta[,i]))/sum(abs(lasso.fit.0$beta))
}


ridge.fit.0          =     glmnet(X.os.train, y.os.train, lambda =0, family = "binomial", alpha = 0)
n.lambdas = dim(ridge.fit$beta)[2]
ridge.beta.ratio     = rep(0, n.lambdas)

for (i in 1:n.lambdas) {
  ridge.beta.ratio[i]   =   sqrt(sum((ridge.fit$beta[,i])^2)/sum((ridge.fit.0$beta)^2))
}


#svm.fit.0          =     glmnet(X.os.train, y.os.train, lambda =0, family = "binomial", alpha = 0)
#n.lambdas = dim(svm.fit$beta)[2]
#svm.beta.ratio     = rep(0, n.lambdas)
#svm.beta.ratio[1] =   sqrt(sum((svm.fit$beta[,1])^2)/sum((svm.fit.0$beta)^2))


eror           =     data.frame(c(rep("lasso", length(lasso.beta.ratio)),  rep("ridge", length(ridge.beta.ratio)) ), 
                                c(lasso.beta.ratio, ridge.beta.ratio) ,
                                c(lasso.cv$cvm, ridge.cv$cvm),
                                c(lasso.cv$cvsd, ridge.cv$cvsd))
colnames(eror) =     c("method", "ratio", "cv", "sd")

eror.plot      =     ggplot(eror, aes(x=ratio, y = cv, color=method)) +   geom_line(size=1) 
eror.plot      =     eror.plot  + scale_x_log10()#(breaks = c(seq(0.1,2.4,0.2)))   
eror.plot      =     eror.plot  + theme(legend.text = element_text(colour="black", size=16, face="bold", family = "Courier")) 
eror.plot      =     eror.plot  + geom_pointrange(aes(ymin=cv-sd, ymax=cv+sd),  size=0.8,  shape=15)
eror.plot      =     eror.plot  + theme(legend.title=element_blank()) 
eror.plot      =     eror.plot  + scale_color_discrete(breaks=c("lasso", "ridge"))
eror.plot      =     eror.plot  + theme(axis.title.x = element_text(size=24),
                                        axis.text.x  = element_text(angle=0, vjust=0.5, size=14),
                                        axis.text.y  = element_text(angle=0, vjust=0.5, size=14)) 
#eror.plot      =     eror.plot  + theme(axis.title.y = element_text(size=16, face="bold", family = "Courier")) 
#eror.plot      =     eror.plot  + xlab( expression(paste( lambda))) + ylab("")
eror.plot      =     eror.plot  + theme(plot.title = element_text(hjust = 0.5, vjust = -10, size=20, family = "Courier"))
#eror.plot      =     eror.plot  + ggtitle(TeX(sprintf("$n$=%s,$p$=%s,$t_{LO}$=%s,$t_{ALO}$=%0.3f,$t_{FIT}$=%.3f",n,p,time.lo,time.alo,time.fit))) 
#eror.plot      =     eror.plot  + ggtitle((sprintf("lasso.cv:%0.3f(sec), lasso.fit:%0.3f(sec) \n ridge.cv:%0.3f(sec), ridge.fit:%0.3f(sec)",time.lasso, time.ridge))) 

eror.plot






################################### 10-fold cv curve: lasso, ridge  ##################################
################################### 10-fold cv curve: lasso, ridge  ##################################

X = X.os.train
y = y.os.train

nfold         =    10
m             =    25
lasso.cv      =    cv.glmnet(X, y, family = "binomial", alpha = 1,  intercept = TRUE, standardize = FALSE,  nfolds = nfold, type.measure="class")
lam.lasso     =    exp(seq(log(max(lasso.cv$lambda)),log(0.00001), (log(0.00001) - log(max(lasso.cv$lambda)))/(m-1)))

ptm           =     proc.time()
lasso.cv      =     cv.glmnet(X, y, lambda = lam.lasso, family = "binomial", alpha = 1,  intercept = TRUE, standardize = FALSE,  nfolds = nfold, type.measure="class")
ptm           =     proc.time() - ptm
time.lasso.cv =     ptm["elapsed"] 

ptm           =     proc.time()
lasso.fit     =     glmnet(X, y, lambda = lasso.cv$lambda, family = "binomial", alpha = 1,  intercept = TRUE, standardize = FALSE)
ptm           =     proc.time() - ptm
time.lasso.fit=     ptm["elapsed"] 

lasso.fit.0   =    glmnet(X, y, lambda = 0, family = "binomial", alpha = 1,  intercept = TRUE, standardize = FALSE)
n.lambdas     =    dim(lasso.fit$beta)[2]

lasso.beta.ratio    =    rep(0, n.lambdas)
for (i in 1:n.lambdas) {
  lasso.beta.ratio[i]   =   sum(abs(lasso.fit$beta[,i]))/sum(abs(lasso.fit.0$beta))
}




ridge.cv      =    cv.glmnet(X, y, family = "binomial", alpha = 0,  intercept = TRUE, standardize = FALSE,  nfolds = nfold, type.measure="class")
lam.ridge     =    exp(seq(log(max(ridge.cv$lambda)),log(0.00001), -(log(max(ridge.cv$lambda))-log(0.00001))/(m-1)))

ptm                 =     proc.time()
ridge.cv            =     cv.glmnet(X, y, lambda = lam.ridge, family = "binomial", alpha = 0,  intercept = TRUE, standardize = FALSE,  nfolds = nfold, type.measure="class")
ptm                 =     proc.time() - ptm
time.ridge.cv       =     ptm["elapsed"] 

ptm                 =     proc.time()
ridge.fit           =     glmnet(X, y, lambda = ridge.cv$lambda, family = "binomial", alpha = 0,  intercept = TRUE, standardize = FALSE)
ptm                 =     proc.time() - ptm
time.ridge.fit      =     ptm["elapsed"] 


ridge.fit.0         =    glmnet(X, y, lambda = 0, family = "binomial", alpha = 0,  intercept = TRUE, standardize = FALSE)


n.lambdas     =    dim(ridge.fit$beta)[2]
ridge.beta.ratio    =    rep(0, n.lambdas)
for (i in 1:n.lambdas) {
  ridge.beta.ratio[i]   =   sqrt(sum((ridge.fit$beta[,i])^2)/sum((ridge.fit.0$beta)^2))
}

eror           =     data.frame(c(rep("lasso", length(lasso.beta.ratio)),  rep("ridge", length(ridge.beta.ratio)) ), 
                                c(lasso.beta.ratio, ridge.beta.ratio) ,
                                c(lasso.cv$cvm, ridge.cv$cvm),
                                c(lasso.cv$cvsd, ridge.cv$cvsd))
colnames(eror) =     c("method", "ratio", "cv", "sd")

eror.plot      =     ggplot(eror, aes(x=ratio, y = cv, color=method)) +   geom_line(size=1) 
eror.plot      =     eror.plot  + scale_x_log10()#(breaks = c(seq(0.1,2.4,0.2)))   
eror.plot      =     eror.plot  + theme(legend.text = element_text(colour="black", size=16, face="bold", family = "Courier")) 
eror.plot      =     eror.plot  + geom_pointrange(aes(ymin=cv-sd, ymax=cv+sd),  size=0.8,  shape=15)
eror.plot      =     eror.plot  + theme(legend.title=element_blank()) 
eror.plot      =     eror.plot  + scale_color_discrete(breaks=c("lasso", "ridge"))
eror.plot      =     eror.plot  + theme(axis.title.x = element_text(size=24),
                                        axis.text.x  = element_text(angle=0, vjust=0.5, size=14),
                                        axis.text.y  = element_text(angle=0, vjust=0.5, size=14)) 
#eror.plot      =     eror.plot  + theme(axis.title.y = element_text(size=16, face="bold", family = "Courier")) 
#eror.plot      =     eror.plot  + xlab( expression(paste( lambda))) + ylab("")
eror.plot      =     eror.plot  + theme(plot.title = element_text(hjust = 0.5, vjust = -10, size=20, family = "Courier"))
#eror.plot      =     eror.plot  + ggtitle(TeX(sprintf("$n$=%s,$p$=%s,$t_{LO}$=%s,$t_{ALO}$=%0.3f,$t_{FIT}$=%.3f",n,p,time.lo,time.alo,time.fit))) 
eror.plot      =     eror.plot  + ggtitle((sprintf("lasso.cv:%0.3f(sec), lasso.fit:%0.3f(sec) \n ridge.cv:%0.3f(sec), ridge.fit:%0.3f(sec)",time.lasso.cv,time.lasso.fit,time.ridge.cv,time.ridge.fit))) 

eror.plot








