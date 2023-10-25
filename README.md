# credit-risk-classification

In this challenge, I used supervised learning methods to train and evaluate a model based on loan risk. The dataset has a historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

The tasks are subdivide into three:

### Split the Data into Training and Testing Sets

1. Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.

2. Created the labels set (y) from the “loan_status” column, and created the features (X) DataFrame from the remaining columns.

3. Split the data into training and testing datasets by using train_test_split.

### Create a Logistic Regression Model with the Original Data

1. Fitted a logistic regression model by using the training data (X_train and y_train).

2. Saved the predictions for the testing data labels by using the testing feature data (X_test) and the fitted model.

3. Evaluated the model’s performance by:
        Generated a confusion matrix and
        Printed the classification report.

4. How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels? <br>

Ans The logistic regression model is 95% accurate at predicting healthy vs high-risk loan labels. Based on the classification report, the precision for healthy loans (label 0) is 1.00, the recall is 0.99,      and the f1-score is 1.00, meaning that the model is highly accurate and it performs well. Whereas, the precision for high-risk loans (label 1) is 0.85, the recall is 0.91, and the f1-score is 0.88 which        means the model is reasonable performing well. Overall, the model shows strong predictive capabilities for both classes, but it's slightly better at predicting healthy loans, which is expected given the 
imbalanced nature of the data.

### A Credit Risk Analysis Report

#### Overview of the analysis

In order to predict the chances of a loan being healthy or at risk, these are the factors considered in the analysis:
* Size of the loan 
* Interest rate 
* Borrower's income
* Debt to income ratio
* Number of accounts the borrower held
* Derogatory marks against the borrower
* Total debt
* Loan status

Furthermore, steps included: 

#### 1) Data Distribution:
   The dataset shows a significant imbalance between healthy and at-risk loans, with 75,036 loans classified as "healthy" (0) and 2,500 loans as "at-risk" (1).

#### 2) Data Preparation: 
The analysis began with data separation, where the data was split into target (y) and label (x) variables. Subsequently, the sklearn train_test_split function was employed to create training and testing data sets, using a random state of 1 for consistency.

#### 3) Model Training: 
The Logistic Regression function from the sklearn linear_model module was used to train a machine learning model on the training data. Predictions were generated based on this model and applied to the testing data.

#### 4) Model Evaluation: 
Model performance was assessed using common metrics. The accuracy score, confusion matrix, and classification report were generated using the sklearn library with the metrics module.

#### 5) Addressing Data Imbalance: 
Recognizing the data imbalance, an additional learning model from imblearn (imbalanced learning) was deployed. The RandomOverSampler function was utilized to resample both the x and y data to address the imbalance.

#### 6) Re-evaluation: 
Following resampling, the Logistic Regression model was again employed to evaluate the results of the resampled data and assess the impact on model performance.

#### Results
* Machine learning model 1

Accuracy: Our model demonstrates a strong accuracy level of 0.99, correctly categorizing around 99% of all loans in our dataset. Nevertheless, it's crucial to be cautious about relying solely on high accuracy when dealing with imbalanced datasets, necessitating consideration of other evaluation metrics.

Precision: Our model exhibits a precision of approximately 0.85. This signifies that when our model identifies a loan as "high-risk," it is accurate approximately 85% of the time. In simpler terms, the model maintains a relatively low false positive rate, rendering it fairly dependable when predicting a loan's risk.

Recall (Sensitivity): Our model's recall stands at around 0.88. This implies that our model is proficient at capturing 89% of the actual high-risk loans within the dataset. It boasts a relatively low false negative rate, indicating that it doesn't overlook many high-risk loans.

* Machine learning model 2

Accuracy: Our secondary model boasts an exceptionally high accuracy of 0.994, signifying that it accurately classifies approximately 99.4% of all loans within our dataset. This exceptional accuracy rate underscores the model's outstanding overall performance.

Precision: The precision of our secondary model stands at around 0.994. This implies that when our model identifies a loan as "high-risk," it is accurate approximately 99.4% of the time. The model maintains an exceedingly low false positive rate, which reinforces its high reliability in flagging loans as risky.

Recall (Sensitivity): The recall of our secondary model is approximately 0.994. This demonstrates that our model effectively captures 99.4% of the actual high-risk loans in the dataset. It achieves an extremely low false negative rate, indicating its rare instances of missing high-risk loans.


#### Summary

The secondary model surpasses the primary model in terms of accuracy, precision, and recall. It attains nearly flawless scores across these three metrics, highlighting its exceptional capability in distinguishing between "healthy" and "high-risk" loans. When we compare the results with those of the primary model, it's evident that this improved performance is a direct outcome of achieving data balance.

While the initial model did exhibit a respectable level of accuracy and remains effective in identifying risky loans, the secondary model's near-perfect precision and recall values for class 1 establish its exceptional reliability, making it a top choice for minimizing false negatives, or instances where it misses identifying high-risk loans.
