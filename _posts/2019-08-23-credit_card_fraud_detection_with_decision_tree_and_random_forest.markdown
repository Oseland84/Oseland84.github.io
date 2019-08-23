---
layout: post
title:      "**Credit Card Fraud Detection  with Decision Tree and Random Forest**"
date:       2019-08-23 16:51:23 +0000
permalink:  credit_card_fraud_detection_with_decision_tree_and_random_forest
---


**Context**

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

**Objective**

To build a model on the ('creditcard.csv') dataset that can help to accurately predict potentially fraudulent credit card transactions moving forward into the future.

**Content**

The datasets contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependent cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

After importing  the necissary libraries, I began as always, by importing and inspecting my dataset.

# importing necessary libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```

```
df = pd.read_csv('creditcard.csv')
df.head(30)
```
The Data was in very good condition, So I will skip most of the ispecting and cleaning process. I did go aghead and scale the "Amount" column, to be more inline with the rest of the data, which did give me a very small bump in model performance. 

```
from sklearn.preprocessing import StandardScaler
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
```

I also thought it was important to take a look at the distribution of the various features.

```
df.hist(figsize=(20,20))
plt.show()
```
 "Hist plot of features"
 
It looks as though all the "V" columns are clustered pretty tightly around 0, with some small amount of variance. We knew going in that this was going to be a highly unbalanced dataset, a quick look at the histogram for Class gives us a nice visual for how severe this actually is. To make sure I am being explicit, here are a couple of print statements that also summarize this imbalance.

```
fraud = len(df[df.Class ==1])
print(f'There are {fraud} cases of fraudulent transactions.')
```
> There are 492 cases of fraudulent transactions.    

```
not_fraud = len(df[df.Class ==0])
print(f'There are {not_fraud} cases of non-fraudulent transactions.')
```
> There are 284315 cases of non-fraudulent transactions.

```
outlier_fraction = fraud / not_fraud
print(f'The percentage of fraudulent cases is only {outlier_fraction}%.')
```
> The percentage of fraudulent cases is only 0.0017304750013189597%.

I want to make a quick correlation matrix as well.This will help to get a feel for what features may be strong predictors as well as if we may need to remove any features that are too strongly correlated to one another.

```
corrmat = df.corr()
fig = plt.figure(figsize=(12,9))

sns.heatmap(corrmat)
plt.show()
```
"Correlation Heatmap"

The vast majority of this heat map shows values of 0, which is to say that it does not appear that any of our "V"(V1-V28) parameters are very correlated. What I really want to focus on here is Class. lighter colors imply a positive correlation, while darker grids imply a negative one. we can see that Class has a noticeable positive correlation to V11, and a very noticeable negative correlation to V12, V14, and especially V17. There doesn't seem to be much correlation with Time, or Amount (our only other labeled data). I'm not sure how this will factor in just yet with our modeling, but it still seems like significant information to possess.

It's time to make a **Decision Tree**! 

To begin, establish predictor variables and target, X and y:

```
X = df.drop('Class',axis=1)
X.head() # make sure it worked
```

```
y = df['Class']
y.head() # make sure it worked
```
Create a training and test set.

```
# importing necessary libraries
from sklearn.model_selection import train_test_split
```

`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)`

```
# importing necessary libraries
from sklearn.tree import DecisionTreeClassifier
```
Instantiate an instance of a single decision tree model.

`dtree = DecisionTreeClassifier()`

Fit the data to that model.

```
# takes a little time
dtree.fit(X_train,y_train)
```
`predictions = dtree.predict(X_test)`

`from sklearn.metrics import classification_report,confusion_matrix`

`print(confusion_matrix(y_test,predictions))`

```
[[93786    52]
 [   30   119]]
```
 
![](https://3.bp.blogspot.com/--jLXutUe5Ss/VvPIO6ZH2tI/AAAAAAAACkU/pvVL4L-a70gnFEURcfBbL_R-GnhBR6f1Q/s1600/ConfusionMatrix.png) 

Let's break this down a little bit before moving on. In this instance (for this project), negative refers to valid transactions, while positive represents fraudulent transactions. This means that a TRUE positive is a prediction of fraud that was predicted correctly, and a FALSE positive is where the model predicted fraud, but it was actually a valid transaction.

Precision – What percent of your predictions were correct? Precision = TP/(TP + FP)

Recall – What percent of the positive cases did you catch? Recall = TP/(TP+FN) "labeled sensitivity-above"

F1 score – What percent of positive predictions were correct? F1 Score = 2*(Recall * Precision) / (Recall + Precision)

`print(classification_report(y_test,predictions))`

```
precision    recall  f1-score   support

           0       1.00      1.00      1.00     93838
           1       0.70      0.80      0.74       149

   micro avg       1.00      1.00      1.00     93987
   macro avg       0.85      0.90      0.87     93987
weighted avg       1.00      1.00      1.00     93987
```

Now let's try with **Random Forest**
Now let us try again with random forest classification (makes use of multiple decision trees instead of just one).

a quick explanation for those who don't already know:

A decision tree is built on an entire dataset, using all the features/variables of interest, whereas a random forest randomly selects observations/rows and specific features/variables to build multiple decision trees from, and then averages the results. It is whats known as an ensemble method.

The workflow process is very similar:

`from sklearn.ensemble import RandomForestClassifier`
`rfc = RandomForestClassifier()`
`rfc.fit(X_train,y_train)`
`rfc_pred = rfc.predict(X_test)`

`print(confusion_matrix(y_test,rfc_pred))`
```
[[93828    10]
 [   28   121]]
```

`print(classification_report(y_test,rfc_pred))`
```
 precision    recall  f1-score   support

           0       1.00      1.00      1.00     93838
           1       0.92      0.81      0.86       149

   micro avg       1.00      1.00      1.00     93987
   macro avg       0.96      0.91      0.93     93987
weighted avg       1.00      1.00      1.00     93987
```
Again, let's break this down a bit more. Valid transactions (0) scored pretty much perfectly across all measures of the classification report, not surprising since our training dataset had 93,838 data points to train on. What we really want to focus on is the scores for our fraudulent transactions (1). As you can see, our Random Forest model scored significantly higher in precision, moderately better on the F-1 score, and slightly better for recall than our single Decision Tree. While our Decision Tree correctly identified 119 cases of fraud, it mistakenly labeled 52 valid transactions as being fraudulent. Our Random Forest model was able to accurately predict 121 cases of fraud (2 more than D.T), but only mislabeled 10, hence the much better precision score, and slightly better F-1 score. In this instance, Recall is only really concerned with how many TRUE positives it was able to predict, and our random forest got 2 more than our Decision Tree. So our Random Forest did better across the board, which is what we hoped for.

From this point I started playing around with the different features of Random Forest, like adjusting the number of trees it would create `n_estimators=` as well as many other adjustments that can be made. In the end, my best fitting model for the time being looks like this:

```
# Random Forrest with n_estimators = 100 
print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(classification_report(y_test,rfc_pred))
```
```
[[93831     7]
 [   28   121]]


              precision    recall  f1-score   support

           0       1.00      1.00      1.00     93838
           1       0.95      0.81      0.87       149

   micro avg       1.00      1.00      1.00     93987
   macro avg       0.97      0.91      0.94     93987
weighted avg       1.00      1.00      1.00     93987
```
**Conclusions/Recomendations**

Let's break this down one last time . Valid transactions (0) scored pretty much perfectly across all measures of the classification report, not surprising since our training dataset had 93,838 data points to train on. What we really want to focus on is the scores for our fraudulent transactions (1). As you can see, our Random Forest model scored significantly higher in precision, moderately better on the F-1 score, and slightly better for recall than our single Decision Tree. While our Decision Tree correctly identified 119 cases of fraud, it mistakenly labeled 52 valid transactions as being fraudulent. Our Random Forest model, in the end was able to accurately predict 121 cases of fraud (2 more than D.T), but only mislabeled 7, giving us a precision score of 95%, and slightly better F-1 score as well. In this instance, Recall is only really concerned with how many TRUE positives it was able to predict, and our random forest got 2 more than our Decision Tree. So our Random Forest did better across the board, which is what we hoped for.

In the end, I was able to create a model that can accurately predict fraudulent credit card transactions with 95% accuracy. While I am fairly pleased with these results, I believe there is still room for improvement. I only ended up using two different models to tune on this dataset, but there are still more options available. While some of them I had to pass over because of computational costs, like KNN (this is a big dataset), I would still be interested in trying SVM (Support Vector Machines) in the future when I have more time. For now, I feel that the models I chose to work with were the best choice for the type of classification I was doing, especially on such a large, lopsided dataset, as they (especially Random Forest) tend to be more forgiving of this. I would recommend continuing to gather data, so that we are able to keep improving the accuracy of our models. I would also recommend trying to focus more on raising the recall score (specifically), even if it hurts some of the other scores. While I really focused on getting the best performing model overall, I would imagine that the most costly mistake that could be made was to miss fraudulent charges initially and be responsible for paying that money back to the customer. However, tweaking the model too much, so that it starts labeling a lot of valid transactions as fraudulent could also be a problem if you start harassing your customers too much about whether or not the charges made where valid.


