# Setting up working directory
setwd('/Users/Miller/Desktop/ML/')
#Importing training file with replacing blank with NA
a=read.csv("pml-training.csv",na.strings=c('','NA'))
b=a[,!apply(a,2,function(x) any(is.na(x)) )]
#Removing variables from 1 to 7
c=b[,-c(1:7)]
#Creating training and test data
inTrain=createDataPartition(y=c$classe, p=0.7, list=FALSE)
training=c[inTrain,]
testing=c[-inTrain, ]
dim(training);dim(testing)
#Building a model with random forest
model=randomForest(classe~., data=training, method='class')
#Predicting a model
pred=predict(model,testing, type='class')
#Prediction results are written in matrix
z=confusionMatrix(pred,testing$classe)
save(z,file='test.RData')
load('test.RData')
z$table
z$overall[1]
#Importing test file with replacing blank with NA
d=read.csv('pml-testing.csv',na.strings=c('','NA'))
e=d[,!apply(d,2,function(x) any(is.na(x)) )]
f=e[,-c(1:7)]
#Predicting model with test file
predicted=predict(model,f,type='class')
save(predicted,file='predicted.RData')
load('predicted.RData')

