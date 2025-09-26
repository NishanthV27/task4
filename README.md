This project demonstrates the use of Logistic Regression to classify breast cancer tumors as malignant or benign using the Breast Cancer dataset from scikit-learn. Logistic Regression is a popular binary classification algorithm that predicts the probability of a class using the sigmoid function.

The workflow includes:

Loading the dataset: 569 samples with 30 features representing tumor characteristics.

Data preprocessing: Splitting into training and test sets (80-20 split) and standardizing features for better model performance.

Training the model: Using LogisticRegression from sklearn to fit the training data.

Evaluating the model: Using confusion matrix, precision, recall, and ROC-AUC to measure model performance.

Threshold tuning: Adjusting the decision threshold to improve recall or precision depending on the use case.

Sigmoid function visualization: Explaining how logistic regression converts linear predictions into probabilities.

Features

Confusion Matrix for model performance

Precision and Recall scores

ROC-AUC score and ROC Curve visualization

Adjustable classification threshold for optimal results

Sigmoid function plot for intuitive understanding
