# Practical Machine Learning - Human Activity Recognition (HAR)
Sohail Iqbal  
Sunday, July 26, 2015  

## Introduction
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self-movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Project Goal
The goal of this project is to predict the manner in which the people did the exercises, which is defined in the "classe" variable in the training dataset. The goal is also describing how the prediction model is built, how it is cross validated, evaluation of the expected out of sample error, and explaining the reasons of the choices made to build this model. The prediction model will be used to predict 20 different test cases.

## Datasets
Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions, and the data was recorded for evaluation purposes, the 5 classes of their exercise activity are:

A: Exactly according to the specification

B: Throwing the elbows to the front 

C: Lifting the dumbbell only halfway

D: Lowering the dumbbell only halfway

E: Throwing the hips to the front

### Data Source
The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv 
The test data are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv 
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 


```r
library(caret)
library(randomForest)
```

### Data Wrangling (Loading & Preprocessing)
Here we load the above downloaded datasets from "pml-training.csv" and "pml-testing.csv" in the R data frames. We replace the character values of NA, NaN, #DIV/0!, or blank values with R's built-in NA type when loading the data, so it can be handled by the corresponding methods.


```r
# pointing to project folder
setwd("~/R/Projects/Practical Machine Learning - HAR/")

# loading raw data
trainingData <- read.csv("pml-training.csv", na.strings=c("NA","NaN","#DIV/0!", ""))
testingData <- read.csv("pml-testing.csv", na.strings=c("NA","NaN","#DIV/0!", ""))

# check loaded data
dim(trainingData)
```

```
## [1] 19622   160
```

```r
dim(testingData)
```

```
## [1]  20 160
```

After simple exploration, we found out that several columns of the raw data set do not contain any values, so we will remove those columns. We will also delete the first 7 columns, namely X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, and num_window. Since columns have no relevance on impact on the prediction outcome.

We also remove the columns with more than 60% of missing values to reduce the noise in the data.


```r
#removing unusable columns
trainingData <- trainingData[,-nearZeroVar(trainingData)]
trainingData <- trainingData[,-c(1:7)]
trainingData$classe <- as.factor(trainingData$classe)

# removing columns with 60%+ NAs
tempData<-colSums(is.na(trainingData)) <= 0.6*nrow(trainingData)
trainingData<-trainingData[, tempData]
rm(tempData)

# making final testing dataset dimensions same as training data
testingData <- testingData[, names(testingData) %in% names(trainingData)]

# checking final datasets
dim(trainingData)
```

```
## [1] 19622    52
```

```r
dim(testingData)
```

```
## [1] 20 51
```

### Data Preparation for Testing and Training
Here we will partition the data for training and testing the models. We will use the 60% of the training dataset for training the prediction model and 40% to test the model. We will then use the original testing dataset from "pml-testing.csv" to make the predictions.

```r
inTrain <- createDataPartition(y=trainingData$classe, p=0.6, list=FALSE)
subsetTraining <- trainingData[inTrain,]
subsetTesting <- trainingData[-inTrain,]
```

## Training The Model using Random Forests
Since this is a classification problem, the Classification Tree and Random Forest are the candidates to best predicts the outcomes from this data. After some initial testing we opted for Random Forrest algorithm as the accuracy rate of this algorithm was way better than Classification Tree method. Following model trains on the testing data subset after running through a 500 trees with 3 predictors each time.

```r
set.seed(112233)
modelFit <- train(classe ~ ., method="rf", data=subsetTraining,
                  ntree=500, tuneGrid=data.frame(.mtry = 3))
modelFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, ntree = 500, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 3
## 
##         OOB estimate of  error rate: 0.82%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3346    2    0    0    0 0.0005973716
## B   18 2251   10    0    0 0.0122860904
## C    0   22 2031    1    0 0.0111976631
## D    0    0   36 1892    2 0.0196891192
## E    0    0    0    5 2160 0.0023094688
```

```r
modelFit
```

```
## Random Forest 
## 
## 11776 samples
##    51 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## Resampling results
## 
##   Accuracy   Kappa      Accuracy SD  Kappa SD   
##   0.9863754  0.9827583  0.002340445  0.002957778
## 
## Tuning parameter 'mtry' was held constant at a value of 3
## 
```

### Accuracy of the Algorithm
Here we will evaluate the performance of the model on the validation data set here called (subsetTesting). A Confusion Matrix, the estimated accuracy and the estimated out-of-sample error of the model are calculated.

```r
predictedData <- predict(modelFit, subsetTesting)
confusionMatrix(subsetTesting$classe, predictedData)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2230    2    0    0    0
##          B   11 1502    5    0    0
##          C    0   15 1351    2    0
##          D    0    0   32 1253    1
##          E    0    0    0    4 1438
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9908          
##                  95% CI : (0.9885, 0.9928)
##     No Information Rate : 0.2856          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9884          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9951   0.9888   0.9733   0.9952   0.9993
## Specificity            0.9996   0.9975   0.9974   0.9950   0.9994
## Pos Pred Value         0.9991   0.9895   0.9876   0.9743   0.9972
## Neg Pred Value         0.9980   0.9973   0.9943   0.9991   0.9998
## Prevalence             0.2856   0.1936   0.1769   0.1605   0.1834
## Detection Rate         0.2842   0.1914   0.1722   0.1597   0.1833
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9974   0.9931   0.9854   0.9951   0.9993
```

```r
postResample(predictedData, subsetTesting$classe)
```

```
##  Accuracy     Kappa 
## 0.9908233 0.9883904
```

## Testing The Model on Test Dataset
Now we have a trained model with 98%+ accuracy, we will use this model to predict the outcome on the given raw testing data from the file "pml-testing.csv".

```r
# final prediction on original testing set (20 samples)
predictTestingData <- predict(modelFit, testingData)

# adding the predcted class to the testing data set (for reference)
testingData$classe <- predictTestingData

#final output
predictTestingData
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
