library(corrplot)
library(ppcor)
library(dplyr)
library(GGally)
library(tseries)
library(purrr)
library(tidyr)
library(readxl)
library(recipes)
library(mlr)
library(mlbench)
library(e1071)
library(kknn)
library(rpart)
library(kernlab)
library(nnet)
library(unbalanced)
library(DiscriMiner)
library(FSelectorRcpp)
library(praznik)
library(randomForest)
library(ada)

bd <- read_excel(".../default of credit card clients.xls", skip = 1)
bd <- bd[,-1]
View(bd)

#ChequearBase
dim(bd)
str(bd)

#valores perdidos
sum(is.na(bd))

#Cambios
#renombrar variables
bd <- bd %>%
rename (BAL = LIMIT_BAL, RPSEP = PAY_0, RPAGO = PAY_2, 
        RPJUL = PAY_3, RPJUN = PAY_4, RPMAY = PAY_5, 
        RPABR = PAY_6, BILLSEP = BILL_AMT1, BILLAGO = BILL_AMT2,
        BILLJUL = BILL_AMT3, BILLJUN = BILL_AMT4, BILLMAY = BILL_AMT5,
        BILLABR = BILL_AMT6, PREPAYSEP = PAY_AMT1, PREPAYAGO = PAY_AMT2,
        PREPAYJUL = PAY_AMT3, PREPAYJUN = PAY_AMT4, PREPAYMAY = PAY_AMT5,
        PREPAYABR = PAY_AMT6, DEFAULT = `default payment next month`)

#sumarle una constante para recategorizar
bd <- bd%>%
  mutate(RPSEP = RPSEP + 1,
         RPAGO = RPAGO + 1,
         RPJUL = RPJUL + 1,
         RPJUN = RPJUN + 1,
         RPMAY = RPMAY + 1,
         RPABR = RPABR + 1)

#recategorizar - dummy encoding
bd <- bd %>%
  mutate(EDUCATION = ifelse(EDUCATION >= 4 | EDUCATION == 0, 4, EDUCATION),
         MARRIAGE = ifelse(MARRIAGE == 0, 3, MARRIAGE))

bd <- bd %>%
  mutate(MUJER = ifelse (SEX == 2, 1, 0), #0 hombre & 1 mujer,
         PREGRADO = ifelse (EDUCATION == 2, 1, 0), 
         HSCHOOL = ifelse (EDUCATION == 3, 1, 0),
         POSGRADO = ifelse (EDUCATION == 1, 1, 0),
         SINGLE = ifelse (MARRIAGE == 2, 1, 0),
         MARRIED = ifelse (MARRIAGE == 1, 1, 0))
         
bd1 <- bd
bd <- bd[,-2:-4]

#recategorizar
bd <- bd %>%
  mutate(RPSEP = ifelse(RPSEP == 0, -1, RPSEP),
         RPAGO = ifelse(RPAGO == 0, -1, RPAGO),
         RPJUL = ifelse(RPJUL == 0, -1, RPJUL),
         RPJUN = ifelse(RPJUN == 0, -1, RPJUN),
         RPMAY = ifelse(RPMAY == 0, -1, RPMAY),
         RPABR = ifelse(RPABR == 0, -1, RPABR))

#Frecuencias
apply(bd[,c(2:4,6:11)],2,table)
summary(bd)

#Factores

bd$SEX <- factor(bd$SEX, levels= 1:2, ordered=TRUE)
bd$EDUCATION <- factor (bd$EDUCATION, levels = 1:4, ordered = TRUE)
bd$MARRIAGE <- factor (bd$MARRIAGE, levels = 1:3, ordered = TRUE)
str(bd)

#Exploratario

#correlación
cor(bd[,-2:-4,])
corrplot(cor(bd[,-2:-4]))
corrplot.mixed(cor(bd[,-2:-4]))

#Asimetría y Curtosis
apply(bd[,c(1,12:23)],2,kurtosis)
apply(bd[,c(1,12:23)],2,skewness)

#Normalidad
apply(bd[,1:20],2,jarque.bera.test)

#Gráficos

ggplot(data = bd) +
  aes(x = BAL)+
  geom_histogram()

ggplot(data = train) +
  aes(x = BILLJUN) +
  geom_histogram()

#Outliers
#usando la distancia euclidiana (si es mayor a 3 es un caso atípico)

bd <- bd%>%
  mutate(out_BAL = abs(scale(bd$BAL)),
        out_SEP = abs(scale(bd$BILLSEP)),
        out_AGO = abs(scale(bd$BILLAGO)),
        out_JUL = abs(scale(bd$BILLJUL)),
        out_JUN = abs(scale(bd$BILLJUN)),
        out_MAY = abs(scale(bd$BILLMAY)),
        out_ABR = abs(scale(bd$BILLABR)))

bd <- bd%>%
  filter(out_BAL < 3 | out_SEP < 3 | out_AGO < 3 | out_JUL < 3 |
           out_JUN < 3, out_MAY < 3, out_ABR <3)

table(bd$DEFAULT)
bd <- bd[,-28:-34]

#Split the dataset

set.seed(100)
index_1 <- sample(1:nrow(bd), round(nrow(bd) * 0.8))
train <- bd[index_1, ]
test  <- bd[-index_1, ]
summary(train)

#Normalizar
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

train <- replace(train, 1:20,(apply(train[,1:20],2,normalize)))

#Balance
Y <- train[,21]
X <- train[,-21]
balance_train <- ubSMOTE(X = X, Y = as.factor(Y$DEFAULT), perc.over = 100, perc.under = 300, k=3)
btrain <- as.data.frame(cbind(X = balance_train$X, DEFAULT = balance_train$Y))
table(btrain$DEFAULT)/nrow(btrain)

##ML-------------------------------------

#task
btrain$DEFAULT <- factor(btrain$DEFAULT, levels = c(0,1))
clasificacion.task <- makeClassifTask(id = "task", data = btrain, target = "DEFAULT", positive = "1")
clasificacion.task
getTaskFeatureNames(clasificacion.task)

##KNN----

#lerner
getParamSet("classif.kknn")
learner.knn <- makeLearner("classif.kknn", 
                          predict.type = "response")
learner.knn$par.set
learner.knn

#Train
mod.knn <- mlr::train(learner.knn, clasificacion.task)
getLearnerModel(mod.knn)

#Predict
predict.knn <- predict(mod.knn, task = clasificacion.task)
head(as.data.frame(predict.knn))
calculateConfusionMatrix(predict.knn)

#Performance
listMeasures(clasificacion.task)
performance(predict.knn, measures = list(acc, mmce, kappa))

#Resampling
RCV.knn <- repcv(learner.knn, clasificacion.task, folds = 3, reps = 2, 
             measures = list(acc, mmce, kappa), stratify = TRUE)

#FSS-Filter
fv.knn <- generateFilterValuesData(clasificacion.task,
                                   method = "FSelectorRcpp_information.gain")
fv.knn$data

lrn.fss.knn <- makeFilterWrapper(learner = "classif.kknn", 
                        fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
rdesc <- makeResampleDesc("RepCV", folds = 3, reps = 2)
r.knn.fss = resample(learner = lrn.fss.knn, task = clasificacion.task, resampling = rdesc, show.info = FALSE, models = TRUE, measures = mlr::acc)
r.knn.fss$aggr

mod.knn.fss <- mlr::train(lrn.fss.knn, clasificacion.task)
predict.knn.fss <- predict(mod.knn.fss, task = clasificacion.task)
performance(predict.knn.fss, measures = list(acc, mmce, kappa))

getLearnerModel(mod.knn)
getLearnerModel(mod.knn.fss)

#FSS-Wraper
lrn.wra.knn <- makeFeatSelWrapper(learner = "classif.kknn",
                                  resampling = rdesc, control = 
                                    makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
r.knn.wra <- resample(lrn.wra.knn, clasificacion.task, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
r.knn.wra$aggr
mod.knn.wra <- mlr::train(lrn.wra.knn, clasificacion.task)
predict.knn.wra <- predict(mod.knn.wra, task = clasificacion.task)
performance(predict.knn.wra, measures = list(acc, mmce, kappa))

##DECISION TREE----

#lerner
getParamSet("classif.rpart")
learner.dt <- makeLearner("classif.rpart", 
                         predict.type = "response")
learner.dt$par.set #same as getparamset

#Train
mod.dt <- mlr::train(learner.dt, clasificacion.task)
getLearnerModel(mod.dt)

#Predict
predict.dt <- predict(mod.dt, task = clasificacion.task)
head(as.data.frame(predict.dt))
calculateConfusionMatrix(predict.dt)

#Performance
listMeasures(clasificacion.task)
performance(predict.dt, measures = list(acc, mmce, kappa))

#Resampling
RCV.dt <- repcv(learner.dt, clasificacion.task, folds = 3, reps = 2, 
             measures = list(acc, mmce, kappa), stratify = TRUE)

#FSS-Filter
lrn.fss.dt <- makeFilterWrapper(learner = "classif.rpart", 
                                 fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
r.dt.fss = resample(learner = lrn.fss.dt, task = clasificacion.task, resampling = rdesc, show.info = FALSE, models = TRUE, measures = mlr::acc)
r.dt.fss$aggr

mod.dt.fss <- mlr::train(lrn.fss.dt, clasificacion.task)
predict.dt.fss <- predict(mod.dt.fss, task = clasificacion.task)
performance(predict.dt.fss, measures = list(acc, mmce, kappa))

getLearnerModel(mod.dt.fss)
getLearnerModel(mod.dt)

#FSS-Wraper
lrn.wra.dt <- makeFeatSelWrapper(learner = "classif.rpart",
                                  resampling = rdesc, control = 
                                    makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
r.dt.wra <- resample(lrn.wra.dt, clasificacion.task, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
r.dt.wra$aggr
mod.dt.wra <- mlr::train(lrn.wra.dt, clasificacion.task)
predict.dt.wra <- predict(mod.dt.wra, task = clasificacion.task)
performance(predict.dt.wra, measures = list(acc, mmce, kappa))
                
##LOGISTIC----

#learner
getParamSet("classif.logreg")
learner.lr <- makeLearner("classif.logreg",
                         predict.type = "response")

#Train
mod.lr <- mlr::train (learner.lr, clasificacion.task)
getLearnerModel(mod.lr)

#Prediction
predict.lr <- predict(mod.lr, clasificacion.task)
calculateConfusionMatrix(predict.lr)

#Performance
performance(predict.lr, measures = list(acc, mmce, kappa))

#Resampling
RCV.lr <- repcv(learner.lr, clasificacion.task, folds = 3, reps = 2, 
                measures = list(acc, mmce, kappa), stratify = TRUE)

#FSS-Filter
lrn.fss.lr <- makeFilterWrapper(learner = "classif.logreg", 
                                fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
r.lr.fss = resample(learner = lrn.fss.lr, task = clasificacion.task, resampling = rdesc, show.info = FALSE, models = TRUE, measures = mlr::acc)
r.lr.fss$aggr

mod.lr.fss <- mlr::train(lrn.fss.lr, clasificacion.task)
predict.lr.fss <- predict(mod.lr.fss, task = clasificacion.task)
performance(predict.lr.fss, measures = list(acc, mmce, kappa))

getLearnerModel(mod.lr.fss)
getLearnerModel(mod.lr)

#FSS-Wraper
lrn.wra.lr <- makeFeatSelWrapper(learner = "classif.logreg",
                                 resampling = rdesc, control = 
                                   makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
r.lr.wra <- resample(lrn.wra.lr, clasificacion.task, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
r.lr.wra$aggr
mod.lr.wra <- mlr::train(lrn.wra.lr, clasificacion.task)
predict.lr.wra <- predict(mod.lr.wra, task = clasificacion.task)
performance(predict.lr.wra, measures = list(acc, mmce, kappa))

##SVM----

#learner
getParamSet("classif.ksvm")
learner.svm <- makeLearner("classif.ksvm",
                          predict.type = "response")

#Train
mod.svm <- mlr::train (learner.svm, clasificacion.task)
getLearnerModel(mod.svm)

#Prediction
predict.svm <- predict(mod.svm, clasificacion.task)
calculateConfusionMatrix(predict.svm)

#Performance
performance(predict.svm, measures = list(acc, mmce, kappa))

#Resampling
RCV.svm <- repcv(learner.svm, clasificacion.task, folds = 3, reps = 2, 
                measures = list(acc, mmce, kappa), stratify = TRUE)
#FSS-Filter
lrn.fss.svm <- makeFilterWrapper(learner = "classif.ksvm", 
                                fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
rdesc.svm <- makeResampleDesc("Holdout")
r.svm.fss = resample(learner = lrn.fss.svm, task = clasificacion.task, resampling = rdesc.svm, show.info = FALSE, models = TRUE, measures = mlr::acc)
r.svm.fss$aggr

mod.svm.fss <- mlr::train(lrn.fss.svm, clasificacion.task)
predict.svm.fss <- predict(mod.svm.fss, task = clasificacion.task)
performance(predict.svm.fss, measures = list(acc, mmce, kappa))

getLearnerModel(mod.svm.fss)
getLearnerModel(mod.svm)

#FSS-Wraper

lrn.wra.svm <- makeFeatSelWrapper(learner = "classif.ksvm",
                                 resampling = rdesc.svm, control = 
                                   makeFeatSelControlRandom(maxit = 1), show.info = FALSE)
r.svm.wra <- resample(lrn.wra.svm, clasificacion.task, resampling = rdesc.svm, models = TRUE, show.info = FALSE, measures = mlr::acc)
r.svm.wra$aggr
mod.svm.wra <- mlr::train(lrn.wra.svm, clasificacion.task)
predict.svm.wra <- predict(mod.svm.wra, task = clasificacion.task)
performance(predict.svm.wra, measures = list(acc, mmce, kappa))

##NB----

#learner
getParamSet("classif.naiveBayes")
learner.nb <- makeLearner("classif.naiveBayes",
                           predict.type = "response")

#Train
mod.nb <- mlr::train(learner.nb, clasificacion.task)
getLearnerModel(mod.nb)

#Prediction
predict.nb <- predict(mod.nb, clasificacion.task)
calculateConfusionMatrix(predict.nb)

#Performance
performance(predict.nb, task = clasificacion.task, measures = list(acc, mmce, kappa), simpleaggr = TRUE)

#Resampling
RCV.nb <- repcv(learner.nb, clasificacion.task, folds = 3, reps = 2, 
             measures = list(acc, mmce, kappa), stratify = TRUE)

#FSS-Filter
lrn.fss.nb <- makeFilterWrapper(learner = "classif.naiveBayes", 
                                 fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
r.nb.fss = resample(learner = lrn.fss.nb, task = clasificacion.task, resampling = rdesc, show.info = FALSE, models = TRUE, measures = mlr::acc)
r.nb.fss$aggr

mod.nb.fss <- mlr::train(lrn.fss.nb, clasificacion.task)
predict.nb.fss <- predict(mod.nb.fss, task = clasificacion.task)
performance(predict.nb.fss, measures = list(acc, mmce, kappa))

getLearnerModel(mod.nb.fss)
getLearnerModel(mod.nb)

#FSS-Wraper

lrn.wra.nb <- makeFeatSelWrapper(learner = "classif.naiveBayes",
                                 resampling = rdesc, control = 
                                   makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
r.nb.wra <- resample(lrn.wra.nb, clasificacion.task, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
r.nb.wra$aggr
mod.nb.wra <- mlr::train(lrn.wra.nb, clasificacion.task)
predict.nb.wra <- predict(mod.nb.wra, task = clasificacion.task)
performance(predict.nb.wra, measures = list(acc, mmce, kappa))

##NN----

#learner
getParamSet("classif.nnet")
learner.nn <- makeLearner("classif.nnet", 
                          predict.type = "response")
learner.nn$par.set

#Train
mod.nn <- mlr::train(learner.nn, clasificacion.task)
getLearnerModel(mod.nn)

#Predict
predict.nn <- predict(mod.nn, task = clasificacion.task)
head(as.data.frame(predict.nn))
calculateConfusionMatrix(predict.nn)

#Performance
performance(predict.nn, measures = list(acc, mmce, kappa))

#Resampling
RCV.nn <- repcv(learner.nn, clasificacion.task, folds = 3, reps = 2, 
             measures = list(acc, mmce, kappa), stratify = TRUE)

#FSS-Filter
lrn.fss.nn <- makeFilterWrapper(learner = "classif.nnet", 
                                fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
r.nn.fss = resample(learner = lrn.fss.nn, task = clasificacion.task, resampling = rdesc, show.info = FALSE, models = TRUE, measures = mlr::acc)
r.nn.fss$aggr

mod.nn.fss <- mlr::train(lrn.fss.nn, clasificacion.task)
predict.nn.fss <- predict(mod.nn.fss, task = clasificacion.task)
performance(predict.nn.fss, measures = list(acc, mmce, kappa))

getLearnerModel(mod.nn.fss)
getLearnerModel(mod.nn)

#FSS-Wraper

lrn.wra.nn <- makeFeatSelWrapper(learner = "classif.nnet",
                                 resampling = rdesc, control = 
                                   makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
r.nn.wra <- resample(lrn.wra.nn, clasificacion.task, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
r.nn.wra$aggr
mod.nn.wra <- mlr::train(lrn.wra.nn, clasificacion.task)
predict.nn.wra <- predict(mod.nn.wra, task = clasificacion.task)
performance(predict.nn.wra, measures = list(acc, mmce, kappa))

##DA----

#learner
getParamSet("classif.linDA")
learner.da <- makeLearner("classif.linDA", 
                          predict.type = "response")
learner.da$par.set

#Train
mod.da <- mlr::train(learner.da, clasificacion.task)
getLearnerModel(mod.da)

#Predict
predict.da <- predict(mod.da, task = clasificacion.task)
head(as.data.frame(predict.da))
calculateConfusionMatrix(predict.da)

#Performance
performance(predict.da, measures = list(acc, mmce, kappa))

#Resampling
RCV.da <- repcv(learner.da, clasificacion.task, folds = 3, reps = 2, 
             measures = list(acc, mmce, kappa), stratify = TRUE)

#FSS-Filter
lrn.fss.da <- makeFilterWrapper(learner = "classif.linDA", 
                                fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
r.da.fss = resample(learner = lrn.fss.da, task = clasificacion.task, resampling = rdesc, show.info = FALSE, models = TRUE, measures = mlr::acc)
r.da.fss$aggr

mod.da.fss <- mlr::train(lrn.fss.da, clasificacion.task)
predict.da.fss <- predict(mod.da.fss, task = clasificacion.task)
performance(predict.da.fss, measures = list(acc, mmce, kappa))

getLearnerModel(mod.nn.fss)
getLearnerModel(mod.nn)

#FSS-Wraper

lrn.wra.da <- makeFeatSelWrapper(learner = "classif.linDA",
                                 resampling = rdesc, control = 
                                   makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
r.da.wra <- resample(lrn.wra.da, clasificacion.task, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
r.da.wra$aggr
mod.da.wra <- mlr::train(lrn.wra.da, clasificacion.task)
predict.da.wra <- predict(mod.da.wra, task = clasificacion.task)
performance(predict.da.wra, measures = list(acc, mmce, kappa))

##Metaclasificadores------

#Bagging----

learner.nn.bagg <- makeLearner("classif.nnet")
learner.nn.bagging <- makeBaggingWrapper(learner.nn.bagg, bw.iters = 20, bw.replace = TRUE)
r.nn.bagging = resample(learner.nn.bagging, clasificacion.task, resampling = rdesc, show.info = FALSE)
r.nn.bagging$aggr

mod.nn.bagg <- mlr::train(learner.nn.bagging, clasificacion.task)
predict.nn.bagg <- predict(mod.nn.bagg, clasificacion.task)
performance(predict.nn.bagg, measures = list(acc, mmce, kappa))

#RandomForest

#learner
getParamSet("classif.randomForest")
learner.randomf <- makeLearner("classif.randomForest", 
                          predict.type = "response", ntree=100)
learner.randomf$par.set

#Train
mod.randomf <- mlr::train(learner.randomf, clasificacion.task)
getLearnerModel(mod.randomf)

#Predict
predict.randomf <- predict(mod.randomf, task = clasificacion.task)
head(as.data.frame(predict.randomf))
calculateConfusionMatrix(predict.randomf)

#Performance
performance(predict.randomf, measures = list(acc, mmce, kappa))

RCV.randomf <- repcv(learner.randomf, clasificacion.task, folds = 3, reps = 2, 
                measures = list(acc, mmce, kappa), stratify = TRUE)

#AdaBoos

#learner
getParamSet("classif.ada")
learner.ada <- makeLearner("classif.ada", 
                               predict.type = "response")
learner.ada$par.set

#Train
mod.ada <- mlr::train(learner.ada, clasificacion.task)
getLearnerModel(mod.ada)

#Predict
predict.ada <- predict(mod.ada, task = clasificacion.task)
head(as.data.frame(predict.ada))
calculateConfusionMatrix(predict.ada)

#Performance
performance(predict.ada, measures = list(acc, mmce, kappa))

RCV.ada <- repcv(learner.ada, clasificacion.task, folds = 3, reps = 2, 
                     measures = list(acc, mmce, kappa), stratify = TRUE)
RCV.ada$aggr

##Benchmark----

lrns <- list(learner.knn, learner.dt, learner.lr,
             learner.svm, learner.nb, learner.nn, learner.da)
rdesc <- makeResampleDesc("RepCV", folds = 3, reps = 2) #Choose the resampling strategy

getBMRPerformances(bmr, as.df = TRUE)
getBMRAggrPerformances(bmr, as.df = TRUE)
