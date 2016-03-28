Practical Machine Learning Project
========================================================

Background
========================================================
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

Data
========================================================
The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

Goal and analysis
========================================================
The goal of this project is determine or predict the manner in wich the partcipants did exercise.

So, first I load the packages for start the study:

```r
library(AppliedPredictiveModeling)
library(caret)
library(ElemStatLearn)
library(pgmm)
library(rpart)
library(randomForest)
library(rattle)
library(ggplot2)
library(e1071)
library(rpart.plot)
```

Obtain the data set, train and test, and fixed a seed.

```r
set.seed(3633)

urlTrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlTest <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

train_data <- read.csv(urlTrain, header = T, na.strings = c("NA", "#DIV/0!", 
    ""))
test_data <- read.csv(urlTest, header = T, na.strings = c("NA", "#DIV/0!", ""))
```

We make a small exploratory data analysis

```r
dim(train_data)
```

```
## [1] 19622   160
```

```r
attributes(train_data)$names
```

```
##   [1] "X"                        "user_name"               
##   [3] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
##   [5] "cvtd_timestamp"           "new_window"              
##   [7] "num_window"               "roll_belt"               
##   [9] "pitch_belt"               "yaw_belt"                
##  [11] "total_accel_belt"         "kurtosis_roll_belt"      
##  [13] "kurtosis_picth_belt"      "kurtosis_yaw_belt"       
##  [15] "skewness_roll_belt"       "skewness_roll_belt.1"    
##  [17] "skewness_yaw_belt"        "max_roll_belt"           
##  [19] "max_picth_belt"           "max_yaw_belt"            
##  [21] "min_roll_belt"            "min_pitch_belt"          
##  [23] "min_yaw_belt"             "amplitude_roll_belt"     
##  [25] "amplitude_pitch_belt"     "amplitude_yaw_belt"      
##  [27] "var_total_accel_belt"     "avg_roll_belt"           
##  [29] "stddev_roll_belt"         "var_roll_belt"           
##  [31] "avg_pitch_belt"           "stddev_pitch_belt"       
##  [33] "var_pitch_belt"           "avg_yaw_belt"            
##  [35] "stddev_yaw_belt"          "var_yaw_belt"            
##  [37] "gyros_belt_x"             "gyros_belt_y"            
##  [39] "gyros_belt_z"             "accel_belt_x"            
##  [41] "accel_belt_y"             "accel_belt_z"            
##  [43] "magnet_belt_x"            "magnet_belt_y"           
##  [45] "magnet_belt_z"            "roll_arm"                
##  [47] "pitch_arm"                "yaw_arm"                 
##  [49] "total_accel_arm"          "var_accel_arm"           
##  [51] "avg_roll_arm"             "stddev_roll_arm"         
##  [53] "var_roll_arm"             "avg_pitch_arm"           
##  [55] "stddev_pitch_arm"         "var_pitch_arm"           
##  [57] "avg_yaw_arm"              "stddev_yaw_arm"          
##  [59] "var_yaw_arm"              "gyros_arm_x"             
##  [61] "gyros_arm_y"              "gyros_arm_z"             
##  [63] "accel_arm_x"              "accel_arm_y"             
##  [65] "accel_arm_z"              "magnet_arm_x"            
##  [67] "magnet_arm_y"             "magnet_arm_z"            
##  [69] "kurtosis_roll_arm"        "kurtosis_picth_arm"      
##  [71] "kurtosis_yaw_arm"         "skewness_roll_arm"       
##  [73] "skewness_pitch_arm"       "skewness_yaw_arm"        
##  [75] "max_roll_arm"             "max_picth_arm"           
##  [77] "max_yaw_arm"              "min_roll_arm"            
##  [79] "min_pitch_arm"            "min_yaw_arm"             
##  [81] "amplitude_roll_arm"       "amplitude_pitch_arm"     
##  [83] "amplitude_yaw_arm"        "roll_dumbbell"           
##  [85] "pitch_dumbbell"           "yaw_dumbbell"            
##  [87] "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" 
##  [89] "kurtosis_yaw_dumbbell"    "skewness_roll_dumbbell"  
##  [91] "skewness_pitch_dumbbell"  "skewness_yaw_dumbbell"   
##  [93] "max_roll_dumbbell"        "max_picth_dumbbell"      
##  [95] "max_yaw_dumbbell"         "min_roll_dumbbell"       
##  [97] "min_pitch_dumbbell"       "min_yaw_dumbbell"        
##  [99] "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"
## [101] "amplitude_yaw_dumbbell"   "total_accel_dumbbell"    
## [103] "var_accel_dumbbell"       "avg_roll_dumbbell"       
## [105] "stddev_roll_dumbbell"     "var_roll_dumbbell"       
## [107] "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   
## [109] "var_pitch_dumbbell"       "avg_yaw_dumbbell"        
## [111] "stddev_yaw_dumbbell"      "var_yaw_dumbbell"        
## [113] "gyros_dumbbell_x"         "gyros_dumbbell_y"        
## [115] "gyros_dumbbell_z"         "accel_dumbbell_x"        
## [117] "accel_dumbbell_y"         "accel_dumbbell_z"        
## [119] "magnet_dumbbell_x"        "magnet_dumbbell_y"       
## [121] "magnet_dumbbell_z"        "roll_forearm"            
## [123] "pitch_forearm"            "yaw_forearm"             
## [125] "kurtosis_roll_forearm"    "kurtosis_picth_forearm"  
## [127] "kurtosis_yaw_forearm"     "skewness_roll_forearm"   
## [129] "skewness_pitch_forearm"   "skewness_yaw_forearm"    
## [131] "max_roll_forearm"         "max_picth_forearm"       
## [133] "max_yaw_forearm"          "min_roll_forearm"        
## [135] "min_pitch_forearm"        "min_yaw_forearm"         
## [137] "amplitude_roll_forearm"   "amplitude_pitch_forearm" 
## [139] "amplitude_yaw_forearm"    "total_accel_forearm"     
## [141] "var_accel_forearm"        "avg_roll_forearm"        
## [143] "stddev_roll_forearm"      "var_roll_forearm"        
## [145] "avg_pitch_forearm"        "stddev_pitch_forearm"    
## [147] "var_pitch_forearm"        "avg_yaw_forearm"         
## [149] "stddev_yaw_forearm"       "var_yaw_forearm"         
## [151] "gyros_forearm_x"          "gyros_forearm_y"         
## [153] "gyros_forearm_z"          "accel_forearm_x"         
## [155] "accel_forearm_y"          "accel_forearm_z"         
## [157] "magnet_forearm_x"         "magnet_forearm_y"        
## [159] "magnet_forearm_z"         "classe"
```

```r
attributes(test_data)$names == attributes(train_data)$names
```

```
##   [1]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [12]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [23]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [34]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [45]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [56]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [67]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [78]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
##  [89]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [100]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [111]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [122]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [133]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [144]  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE  TRUE
## [155]  TRUE  TRUE  TRUE  TRUE  TRUE FALSE
```

Clean data for this column having 90% of NAs will be removed. In the same way the variables X, user, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window and num_window are deleted because they do not provide meaningful information to the analysis.


```r
PSum_var <- (colSums(is.na(train_data))/dim(train_data)[1])
PSum_var
```

```
##                        X                user_name     raw_timestamp_part_1 
##                   0.0000                   0.0000                   0.0000 
##     raw_timestamp_part_2           cvtd_timestamp               new_window 
##                   0.0000                   0.0000                   0.0000 
##               num_window                roll_belt               pitch_belt 
##                   0.0000                   0.0000                   0.0000 
##                 yaw_belt         total_accel_belt       kurtosis_roll_belt 
##                   0.0000                   0.0000                   0.9798 
##      kurtosis_picth_belt        kurtosis_yaw_belt       skewness_roll_belt 
##                   0.9809                   1.0000                   0.9798 
##     skewness_roll_belt.1        skewness_yaw_belt            max_roll_belt 
##                   0.9809                   1.0000                   0.9793 
##           max_picth_belt             max_yaw_belt            min_roll_belt 
##                   0.9793                   0.9798                   0.9793 
##           min_pitch_belt             min_yaw_belt      amplitude_roll_belt 
##                   0.9793                   0.9798                   0.9793 
##     amplitude_pitch_belt       amplitude_yaw_belt     var_total_accel_belt 
##                   0.9793                   0.9798                   0.9793 
##            avg_roll_belt         stddev_roll_belt            var_roll_belt 
##                   0.9793                   0.9793                   0.9793 
##           avg_pitch_belt        stddev_pitch_belt           var_pitch_belt 
##                   0.9793                   0.9793                   0.9793 
##             avg_yaw_belt          stddev_yaw_belt             var_yaw_belt 
##                   0.9793                   0.9793                   0.9793 
##             gyros_belt_x             gyros_belt_y             gyros_belt_z 
##                   0.0000                   0.0000                   0.0000 
##             accel_belt_x             accel_belt_y             accel_belt_z 
##                   0.0000                   0.0000                   0.0000 
##            magnet_belt_x            magnet_belt_y            magnet_belt_z 
##                   0.0000                   0.0000                   0.0000 
##                 roll_arm                pitch_arm                  yaw_arm 
##                   0.0000                   0.0000                   0.0000 
##          total_accel_arm            var_accel_arm             avg_roll_arm 
##                   0.0000                   0.9793                   0.9793 
##          stddev_roll_arm             var_roll_arm            avg_pitch_arm 
##                   0.9793                   0.9793                   0.9793 
##         stddev_pitch_arm            var_pitch_arm              avg_yaw_arm 
##                   0.9793                   0.9793                   0.9793 
##           stddev_yaw_arm              var_yaw_arm              gyros_arm_x 
##                   0.9793                   0.9793                   0.0000 
##              gyros_arm_y              gyros_arm_z              accel_arm_x 
##                   0.0000                   0.0000                   0.0000 
##              accel_arm_y              accel_arm_z             magnet_arm_x 
##                   0.0000                   0.0000                   0.0000 
##             magnet_arm_y             magnet_arm_z        kurtosis_roll_arm 
##                   0.0000                   0.0000                   0.9833 
##       kurtosis_picth_arm         kurtosis_yaw_arm        skewness_roll_arm 
##                   0.9834                   0.9799                   0.9832 
##       skewness_pitch_arm         skewness_yaw_arm             max_roll_arm 
##                   0.9834                   0.9799                   0.9793 
##            max_picth_arm              max_yaw_arm             min_roll_arm 
##                   0.9793                   0.9793                   0.9793 
##            min_pitch_arm              min_yaw_arm       amplitude_roll_arm 
##                   0.9793                   0.9793                   0.9793 
##      amplitude_pitch_arm        amplitude_yaw_arm            roll_dumbbell 
##                   0.9793                   0.9793                   0.0000 
##           pitch_dumbbell             yaw_dumbbell   kurtosis_roll_dumbbell 
##                   0.0000                   0.0000                   0.9796 
##  kurtosis_picth_dumbbell    kurtosis_yaw_dumbbell   skewness_roll_dumbbell 
##                   0.9794                   1.0000                   0.9795 
##  skewness_pitch_dumbbell    skewness_yaw_dumbbell        max_roll_dumbbell 
##                   0.9794                   1.0000                   0.9793 
##       max_picth_dumbbell         max_yaw_dumbbell        min_roll_dumbbell 
##                   0.9793                   0.9796                   0.9793 
##       min_pitch_dumbbell         min_yaw_dumbbell  amplitude_roll_dumbbell 
##                   0.9793                   0.9796                   0.9793 
## amplitude_pitch_dumbbell   amplitude_yaw_dumbbell     total_accel_dumbbell 
##                   0.9793                   0.9796                   0.0000 
##       var_accel_dumbbell        avg_roll_dumbbell     stddev_roll_dumbbell 
##                   0.9793                   0.9793                   0.9793 
##        var_roll_dumbbell       avg_pitch_dumbbell    stddev_pitch_dumbbell 
##                   0.9793                   0.9793                   0.9793 
##       var_pitch_dumbbell         avg_yaw_dumbbell      stddev_yaw_dumbbell 
##                   0.9793                   0.9793                   0.9793 
##         var_yaw_dumbbell         gyros_dumbbell_x         gyros_dumbbell_y 
##                   0.9793                   0.0000                   0.0000 
##         gyros_dumbbell_z         accel_dumbbell_x         accel_dumbbell_y 
##                   0.0000                   0.0000                   0.0000 
##         accel_dumbbell_z        magnet_dumbbell_x        magnet_dumbbell_y 
##                   0.0000                   0.0000                   0.0000 
##        magnet_dumbbell_z             roll_forearm            pitch_forearm 
##                   0.0000                   0.0000                   0.0000 
##              yaw_forearm    kurtosis_roll_forearm   kurtosis_picth_forearm 
##                   0.0000                   0.9836                   0.9836 
##     kurtosis_yaw_forearm    skewness_roll_forearm   skewness_pitch_forearm 
##                   1.0000                   0.9835                   0.9836 
##     skewness_yaw_forearm         max_roll_forearm        max_picth_forearm 
##                   1.0000                   0.9793                   0.9793 
##          max_yaw_forearm         min_roll_forearm        min_pitch_forearm 
##                   0.9836                   0.9793                   0.9793 
##          min_yaw_forearm   amplitude_roll_forearm  amplitude_pitch_forearm 
##                   0.9836                   0.9793                   0.9793 
##    amplitude_yaw_forearm      total_accel_forearm        var_accel_forearm 
##                   0.9836                   0.0000                   0.9793 
##         avg_roll_forearm      stddev_roll_forearm         var_roll_forearm 
##                   0.9793                   0.9793                   0.9793 
##        avg_pitch_forearm     stddev_pitch_forearm        var_pitch_forearm 
##                   0.9793                   0.9793                   0.9793 
##          avg_yaw_forearm       stddev_yaw_forearm          var_yaw_forearm 
##                   0.9793                   0.9793                   0.9793 
##          gyros_forearm_x          gyros_forearm_y          gyros_forearm_z 
##                   0.0000                   0.0000                   0.0000 
##          accel_forearm_x          accel_forearm_y          accel_forearm_z 
##                   0.0000                   0.0000                   0.0000 
##         magnet_forearm_x         magnet_forearm_y         magnet_forearm_z 
##                   0.0000                   0.0000                   0.0000 
##                   classe 
##                   0.0000
```

```r
var_nelim <- NULL
j = 1
for (i in 8:dim(train_data)[2]) {
    if (PSum_var[i] <= 0.9) {
        var_nelim[j] = i
        j = j + 1
    }
}
```

varnelim represents the variables to be considered in the analysis, these variables are shown below

```r
var_nelim
```

```
##  [1]   8   9  10  11  37  38  39  40  41  42  43  44  45  46  47  48  49
## [18]  60  61  62  63  64  65  66  67  68  84  85  86 102 113 114 115 116
## [35] 117 118 119 120 121 122 123 124 140 151 152 153 154 155 156 157 158
## [52] 159 160
```

Select those variables in train_data and test_data

```r
train_data <- train_data[, var_nelim]
test_data <- test_data[, var_nelim]
dim(test_data)
```

```
## [1] 20 53
```

```r
dim(train_data)
```

```
## [1] 19622    53
```

Divide the data, we selected 80% for the training data and 20% for test data

```r
attach(train_data)
inTrain <- createDataPartition(classe, p = 0.8, list = F)
training <- train_data[inTrain, ]
testing <- train_data[-inTrain, ]
dim(training)
```

```
## [1] 15699    53
```

```r
dim(testing)
```

```
## [1] 3923   53
```

We adjust a first model, a tree

```r
fit_data <- rpart(classe ~ ., data = training, method = "class")
pred1 <- predict(fit_data, testing, type = "class")
confusionMatrix(testing$classe, pred1)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 988  35  35  50   8
##          B 116 403 111  55  74
##          C  11  54 565  39  15
##          D  34  57  57 437  58
##          E   8  55  67  50 541
## 
## Overall Statistics
##                                         
##                Accuracy : 0.748         
##                  95% CI : (0.734, 0.761)
##     No Information Rate : 0.295         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.681         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.854    0.667    0.677    0.693    0.777
## Specificity             0.954    0.893    0.961    0.937    0.944
## Pos Pred Value          0.885    0.531    0.826    0.680    0.750
## Neg Pred Value          0.940    0.936    0.917    0.941    0.952
## Prevalence              0.295    0.154    0.213    0.161    0.177
## Detection Rate          0.252    0.103    0.144    0.111    0.138
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.904    0.780    0.819    0.815    0.861
```

The accuracy rate is 0.7545. 

```r
fancyRpartPlot(fit_data)
```

![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9.png) 

The second model a random forest

```r
fitRF <- randomForest(classe ~ ., data = training)
predRF <- predict(fitRF, testing, type = "class")
confusionMatrix(testing$classe, predRF)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    1    0    0    0
##          B    2  757    0    0    0
##          C    0    6  677    1    0
##          D    0    0   12  630    1
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                         
##                Accuracy : 0.994         
##                  95% CI : (0.991, 0.996)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.993         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.991    0.983    0.998    0.999
## Specificity             1.000    0.999    0.998    0.996    1.000
## Pos Pred Value          0.999    0.997    0.990    0.980    1.000
## Neg Pred Value          0.999    0.998    0.996    1.000    1.000
## Prevalence              0.285    0.195    0.176    0.161    0.184
## Detection Rate          0.284    0.193    0.173    0.161    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.999    0.995    0.990    0.997    0.999
```

The accuracy rate is 0.9977

```r
plot(fitRF)
```

![plot of chunk unnamed-chunk-11](figure/unnamed-chunk-11.png) 


```r
varImpPlot(fitRF)
```

![plot of chunk unnamed-chunk-12](figure/unnamed-chunk-12.png) 


Prediction on testing data
========================================================
The accuracy rate of the random forest is greater than the tree, therefore the prediction in the test set will be done with the random forest.


```r
predict(fitRF, newdata = test_data)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

