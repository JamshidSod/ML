---
title: "Practical Machine Learning"
author: "Jamshid Sodikov"
date: "September 24, 2015"
output: html_document
---

_**Background**_

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement ??? a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

_Library and Data Download_

```r
library(caret)
library(rpart)
library(rpart.plot)
library(corrplot)
```





**Read Downloaded Data into data frames such as trainRaw and testRaw**

```r
trainRaw <- read.csv("./specdata/pml-training.csv")
testRaw <- read.csv("./specdata/pml-testing.csv")
dim(trainRaw) # 17138 obs, 160 variables
```

```
## [1] 17138   160
```

```r
dim(testRaw) # 20 obs, 160 varibales
```

```
## [1]  20 160
```

**Data Preprocessing**
Let's get rid of NAs and missing values

```r
sum(complete.cases(trainRaw))
```

```
## [1] 350
```

```r
trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0] 
testRaw <- testRaw[, colSums(is.na(testRaw)) == 0] 
dim(trainRaw) # 17138 obs, 84 variables
```

```
## [1] 17138    84
```

```r
dim(testRaw) # 20 obs, 60 varibales
```

```
## [1] 20 60
```
Now remove unneccesary columns which don't have effect on accelerometer measurements


```r
classe <- trainRaw$classe
trainRemove <- grepl("^X|timestamp|window", names(trainRaw))
trainRaw <- trainRaw[, !trainRemove]
trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testRaw))
testRaw <- testRaw[, !testRemove]
testCleaned <- testRaw[, sapply(testRaw, is.numeric)]
```
Now, the cleaned training data set contains 17138 observations and 44 variables, while the testing data set contains 20 observations and 53 variables. The "classe" variable is still in the cleaned training set.

**Data Partition**
The cleaned training set is seperated into a pure training data set (70%) and a validation data set (30%). We will use the validation data set to conduct cross validation in next steps.


```r
set.seed(123) 
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
```

```
## Warning in createDataPartition(trainCleaned$classe, p = 0.7, list = F):
## Some classes have a single record ( ) and these will be selected for the
## sample
```

```r
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]
```

**Model Building**

GMB algorithm is utilised to develop predictive model for activity recognition. We will use 3-fold cross validation when applying the algorithm.


```r
fitControl <- trainControl(## 3-fold CV
        method = "repeatedcv",
        number = 3,
        ## repeated three times
        repeats = 3)
set.seed(825)
gbmFit1 <- train(classe ~ ., data = trainData,
                 method = "gbm",
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE)


gbmFit1
```

```
## Stochastic Gradient Boosting 
## 
## 11999 samples
##    43 predictor
##     6 classes: '', 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold, repeated 3 times) 
## 
## Summary of sample sizes: 7999, 7999, 8000, 8000, 8000, 7998, ... 
## 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD
##   1                   50      0.7735652  0.7030094  0.006823868
##   1                  100      0.8399035  0.7902697  0.004369422
##   1                  150      0.8720731  0.8324228  0.004282767
##   2                   50      0.8676840  0.8264580  0.007315732
##   2                  100      0.9153825  0.8891917  0.006696787
##   2                  150      0.9371899  0.9178238  0.006267642
##   3                   50      0.9078542  0.8792776  0.006435375
##   3                  100      0.9463294  0.9297939  0.005722229
##   3                  150      0.9599416  0.9476248  0.004957235
##   Kappa SD   
##   0.009025610
##   0.005881494
##   0.005686340
##   0.009778451
##   0.008813261
##   0.008222604
##   0.008495464
##   0.007500500
##   0.006494351
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

Next is to estimate the performance of the model on the validation data set.


```r
predictRf <- predict(gbmFit1, testData)
```


```r
accuracy <- postResample(predictRf, testData$classe)
accuracy
```

```
##  Accuracy     Kappa 
## 0.9575793 0.9445329
```

So, the estimated accuracy of the model is 95.75%. 

**Prediction on the Test Data Set**
Now, we apply the model to the original testing data set downloaded from the data source.


```r
result <- predict(gbmFit1, testCleaned[, -length(names(testCleaned))])
result
```

```
##  [1] B A B A A D D B A A B C B A D B A B B B
## Levels:  A B C D E
```

Correlation Matrix Visualization

```r
corrPlot <- cor(trainData[, -length(names(trainData))])
corrplot(corrPlot, method="color")
```

![plot of chunk unnamed-chunk-11](figure/unnamed-chunk-11-1.png) 
Decision Tree Visualization

```r
treeModel <- rpart(classe ~ ., data=trainData, method="class")
prp(treeModel) 
```

![plot of chunk unnamed-chunk-12](figure/unnamed-chunk-12-1.png) 


**Generating Files to submit as answers for the Assignment:**

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(result)
```


