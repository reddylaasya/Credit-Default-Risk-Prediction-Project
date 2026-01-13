# Credit-Default-Risk-Prediction-Project
Developed a hybrid ML system (LightGBM + Logistic Regression) for credit default prediction with feature engineering, SMOTE balancing, metric evaluation (AUC-ROC, F1, RMSE, R²), and clustering (KMeans/DBSCAN).

## Overview
This project builds a machine learning pipeline for credit default risk estimation, combining both classification and regression. It uses LendingClub loan data to:
1. Predict the probability of loan default
2. Predict the defaulted loan amount
3. Calculate Expected Loss (EL) as:

   Expected Loss = Probability of Default × Predicted Loss Amount
   
This approach creates a practical credit risk estimation framework, useful for financial risk analysis.

## Objectives

* Predict which loans are likely to default
* Estimate monetary delinquency for risky loans
* Compute expected loss for each borrower
* Segment borrowers based on financial behavior
* Visualize model performance and risk patterns

## Dataset Details

Source: LendingClub Accepted Loans (2007–2018Q4)

Key features used: loan amount, term, employment length, home ownership, annual income, purpose, debt-to-income ratio, FICO score range, revolving utilization, and loan status.

Labeling for default:
Statuses considered default include Charged Off, Default, Late (31-120 days), and Does Not Meet Credit Policy (Charged Off).
All remaining statuses are treated as non-default.

## Methodology

Data preprocessing included:
* Numeric imputation for missing values (mean strategy)
* Label encoding of categorical variables
* Transformation of income to logarithmic scale
* FICO score midpoint calculation
* Standard scaling of features

Class imbalance was handled using SMOTE oversampling.

## Models Used

Classification models:
* LightGBM Classifier (primary)
* Logistic Regression (baseline)

Regression models:
* LightGBM Regressor (primary)
* Linear Regression (baseline)

Expected Loss was computed by multiplying predicted probability of default with predicted loss amount.

## Evaluation Metrics

For classification:
* Accuracy
* F1 Score
* ROC-AUC Score
* Precision, Recall, Support (classification report)

For regression:
* RMSE (Root Mean Squared Error)
* R² Score (Explained Variance)

## Clustering Component

Borrower segmentation was performed using KMeans and DBSCAN clustering to identify natural groupings of borrowers. Features used for clustering included loan amount, log-transformed income, FICO scores, DTI ratio, and revolving utilization.

Clustering provides insights into borrower risk categories and helps validate model outcomes.

## Visualizations Included

The project includes the following plots for interpretability:
* Class distribution (Default vs Non-default)
* ROC curve for classification
* Actual vs predicted loss scatter plot
* Expected loss distribution histogram
* KMeans clusters (FICO vs Income)
* DBSCAN clusters (FICO vs Income)

## Technologies Used

Programming Language: Python
Major Libraries: Pandas, NumPy, Scikit-learn, LightGBM, Imbalanced-learn, Seaborn, Matplotlib

## Results Summary

* LightGBM outperformed baseline models in both classification and regression tasks
* Expected Loss metric provides more realistic financial risk estimation than simple default prediction
* Clustering revealed borrower segments based on creditworthiness and income
* Visualizations validated dataset behavior and model effectiveness

## Future Improvements

Potential future enhancements include:
* Hyperparameter tuning (Optuna, Bayesian Optimization)
* SHAP-based model explainability
* Time-series credit cycle modeling
* API deployment using FastAPI or Flask
* Interactive dashboard via Streamlit or Dash
* Basel-compliant LGD and EAD modeling

## Conclusion

This project demonstrates a comprehensive approach to credit risk modeling by combining:
* Classification (Probability of Default)
* Regression (Loss Severity)
* Expected Loss Estimation
* Clustering-based borrower segmentation

Such a system can support banks, credit organizations, and fintech platforms in making informed lending decisions and risk assessments.
