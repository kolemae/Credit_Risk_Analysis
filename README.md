# Credit_Risk_Analysis
Using Python and SciKitLearn to evaluate and predict credit risk.

## Table of contents
* [Resources](#resources)
* [Overview of Loan Prediction Risk](#overview-of-loan-prediction-risk)
* [Results](#results)
* [Summary](#summary)

### Resources
- Data Source: LoanStats_2019Q1.csv
- Tools: Python 3.7.13, Jupyter Notebook

## Overview of Loan Prediction Risk
I applied supervised machine learning to predict credit risk by employing different techniques to train and evaluate models. The purpose of this was to identify and recommend whether they *should* be used to predict credit risk. The models I used were:
- RandomOverSampler
- SMOTE
- ClusterCentroids
- SMOTEENN
- BalancedRandomForestClassifier
- EasyEnsembleClassifier

## Results
First I tried out **oversampling** with RandomOverSampler and SMOTE.

RandomOverSampler                    |  SMOTE
:-----------------------------------:|:-----------------------------------:
![RandomOverSampler](/Images/ROS.png) |  ![SMOTE](/Images/SMOTE.png)

- Both oversampling models perform similarly
  - The accuracy difference is negligible
  - RandomOverSampler has slightly better recall for high-risk loans
  - SMOTE has slightly better recall for low-risk loans
- The precision is low for high-risk loans and is high for low-risk loans

Next I applied **undersampling** with ClusterCentroids.

![CC](/Images/CC.png)

- Undersampling performed worse than oversampling
  - The accuracy difference is negligible
  - RandomOverSampler has slightly better recall for high-risk loans
  - SMOTE has slightly better recall for low-risk loans
- The precision is low for high-risk loans and is high for low-risk loans

## Summary
