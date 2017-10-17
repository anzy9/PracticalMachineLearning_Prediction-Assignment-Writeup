# Prediction Assignment Writeup 
Anjali Singh  
16 October 2017  



## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data
Data

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/ha

Training data
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

Test data
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

## Libraries

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.4.2
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.4.2
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```
## Loading Data

```r
rm(list=ls())  
tlink <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
vlink <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(tlink))
validation<- read.csv(url(vlink))
```
## Data Processing
Removing the near zero variance, NA values. We then removed the first 7 columns as they were not related to the classe variable. We also divided the data into Train and Test set

```r
nzv <- nearZeroVar(training)
training<-training[,-nzv]

#removing predictors with NA values
training<-training[,colSums(is.na(training))==0]
training<-training[,-c(1:7)]
training$classe = factor(training$classe)
#Partition the data into Training and Testing
inTrain <- createDataPartition(y=training$classe, p=0.7, list=F)
training <- training[inTrain, ]
testing <- training[-inTrain, ]
```
## Cross Validation
We will be doing 3 fold cross validation to train our model using first Rtree method and then random
forest method later. The one with better accuracy will be selected the right model technique.

```r
set.seed(9999)
cv = trainControl(method="cv",number=5,allowParallel=TRUE,verboseIter=TRUE)
modelrf = train(classe~., data=training, method="rf",trControl=cv)#Randon forest
modeltree = train(classe~.,data=training,method="rpart",trControl=cv)# RTree
```
Checking for accurancy on training and testing data.
For random forest

```r
#training data
prediction_rf1 <- predict(modelrf,newdata=training)
confusionMatrix(prediction_rf1,training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

```r
#testing data
prediction_rf2 <- predict(modelrf,newdata=testing)
confusionMatrix(prediction_rf2,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1173    0    0    0    0
##          B    0  789    0    0    0
##          C    0    0  723    0    0
##          D    0    0    0  673    0
##          E    0    0    0    0  764
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9991, 1)
##     No Information Rate : 0.2846     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2846   0.1914   0.1754   0.1633   0.1853
## Detection Rate         0.2846   0.1914   0.1754   0.1633   0.1853
## Detection Prevalence   0.2846   0.1914   0.1754   0.1633   0.1853
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```
For RTree

```r
#training data
prediction_tree1 <- predict(modeltree,newdata=training)
confusionMatrix(prediction_tree1,training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3555 1102 1125 1058  587
##          B   56  908   75  381  494
##          C  200  277 1028  297  317
##          D   92  370  168  516  383
##          E    3    1    0    0  744
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4914          
##                  95% CI : (0.4831, 0.4998)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3345          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9101   0.3416  0.42905  0.22913  0.29465
## Specificity            0.6061   0.9092  0.90380  0.91180  0.99964
## Pos Pred Value         0.4787   0.4744  0.48513  0.33748  0.99465
## Neg Pred Value         0.9444   0.8520  0.88225  0.85780  0.86288
## Prevalence             0.2843   0.1935  0.17442  0.16394  0.18381
## Detection Rate         0.2588   0.0661  0.07483  0.03756  0.05416
## Detection Prevalence   0.5407   0.1393  0.15425  0.11131  0.05445
## Balanced Accuracy      0.7581   0.6254  0.66642  0.57046  0.64715
```

```r
#testing data
prediction_tree2 <- predict(modeltree,newdata=testing)
confusionMatrix(prediction_tree2,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1066  334  339  316  184
##          B   19  264   31  119  155
##          C   55   91  305   88   89
##          D   31  100   48  150  115
##          E    2    0    0    0  221
## 
## Overall Statistics
##                                          
##                Accuracy : 0.4867         
##                  95% CI : (0.4713, 0.502)
##     No Information Rate : 0.2846         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.3279         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9088  0.33460  0.42185  0.22288  0.28927
## Specificity            0.6022  0.90279  0.90497  0.91476  0.99940
## Pos Pred Value         0.4761  0.44898  0.48567  0.33784  0.99103
## Neg Pred Value         0.9432  0.85144  0.88037  0.85780  0.86073
## Prevalence             0.2846  0.19141  0.17540  0.16327  0.18535
## Detection Rate         0.2586  0.06405  0.07399  0.03639  0.05361
## Detection Prevalence   0.5432  0.14265  0.15235  0.10771  0.05410
## Balanced Accuracy      0.7555  0.61870  0.66341  0.56882  0.64434
```
The accuracy for Rtree(.59)is less than Random Forest(1). We will use Random forest on validation data

## Validation

```r
nzv <- nearZeroVar(validation)
validation<-validation[,-nzv]

#removing predictors with NA values
validation<-validation[,colSums(is.na(validation))==0]
validation<-validation[,-c(1:7)]

predictionFinal<- predict(modelrf, newdata=validation)
predictionFinal
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
## Conclusion
We got all the prediction right



