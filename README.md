# Credit_Risk_Analysis
Using Python and SciKitLearn to evaluate and predict credit risk.

## Table of contents
* [Overview of Loan Prediction Risk](#overview-of-loan-prediction-risk)
* [Oversampling](#oversampling)
* [Undersampling](#undersampling)
* [Combination](#combination)
* [Ensemble Classifiers](#ensemble-classifiers)
* [Summary](#summary)

### Resources
- Data Source: LoanStats_2019Q1.csv
- Tools: Python 3.7.13, Jupyter Notebook

## Overview of Loan Prediction Risk
I applied supervised machine learning to predict credit risk by employing different techniques to train and evaluate models. The purpose of this was to identify and recommend whether they *should* be used to predict credit risk. **All random states were set to 1 to ensure consistency between tests.** The models I used were:
- RandomOverSampler
- SMOTE
- ClusterCentroids
- SMOTEENN
- BalancedRandomForestClassifier
- EasyEnsembleClassifier

## Results
### Oversampling
First I tried out RandomOverSampler and SMOTE.

RandomOverSampler                    |  SMOTE
:-----------------------------------:|:-----------------------------------:
![RandomOverSampler](/Images/ROS.png) |  ![SMOTE](/Images/SMOTE.png)

- Both oversampling models performed similarly
  - The accuracy difference was negligible
  - RandomOverSampler had slightly better recall for high-risk loans
  - SMOTE had slightly better recall for low-risk loans
- The precision was low for high-risk loans and is high for low-risk loans

### Undersampling
Next I applied ClusterCentroids.

![CC](/Images/CC.png)

- Undersampling performed worse than oversampling
  - The accuracy difference was more than 10% lower compared to oversampling
  - Low-risk loan recall especially decreased
- The precision was low for high-risk loans and is high for low-risk loans, again

### Combination
I then used SMOTEENN.

![SMOTEENN](/Images/SMOTEENN.png)

- SMOTEENN had only slight improvements from previous models
  - The accuracy difference was 1-13% better in comparison to the over and under sampled models
  - Specificity was best in this model
- Precision remained the same as past models

### Ensemble Classifiers
Finally I used BalancedRandomForestClassifier and EasyEnsembleClassifier to predict credit risk.

BalancedRandomForestClassifier       |  EasyEnsembleClassifier
:-----------------------------------:|:-----------------------------------:
![brf](/Images/brf.png) |  ![eec](/Images/eec.png)

- Both ensemble classifiers performed better than other models
- The accuracy was best in the EasyEnsembleClassifier at 93.2%
- The precision remained low for high-risk loans and high for low-risk loans
  - EEC had the highest precision for high-risk loans at 0.09%
  - Both models had 100% precision for low-risk loans

## Summary
For all of the models, high-risk loans had low precision and low-risk loans had high precision. The ensemble classifiers were both able to get to 100% precision for low-risk loans. EasyEnsembleClassifier performed the best, with 93.2% accuracy. All other models had an accuracy score under 80%. For those reasons I would only recommend EasyEnsembleClassifier for use, ***based on these previous models***. However, since high-risk loans cannot be reliably predicted, ***I'd recommend not using any of these models and testing other models to find a better predictor.***
