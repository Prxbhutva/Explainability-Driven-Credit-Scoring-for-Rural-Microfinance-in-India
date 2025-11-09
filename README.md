# Explainability-Driven-Credit-Scoring-for-Rural-Microfinance-in-India
#Project Overview 
This project introduces an Explainability-Driven Credit Scoring Framework for rural microfinance borrowers in India.
Its core objective is to design a fair, interpretable, and data-driven system capable of predicting borrower defaults while ensuring transparency and trust — two critical pillars of financial inclusion.
The project leverages:
1. A semi-synthetic dataset derived from Kaggle’s Rural Credit Data for Microfinance Analysis
2. Optuna for model optimization and hyperparameter tuning
3. XGBoost as the best-performing algorithm
4. SHAP (SHapley Additive exPlanations) for explainability
5. Streamlit for deployment and visualization

#Dataset
Source: [Rural Credit Data for Microfinance Analysis – Kaggle](https://www.kaggle.com/datasets/heydido/creditloan-dataset-rural-india)
Size: ~40,000 borrower records with 21 original socioeconomic and financial features.
Challenge: The dataset was unsupervised (no target variable for default prediction).
Solution: Created a synthetic output column (“default”) using:
Logistic probability function simulating real-world credit behavior.
Bernoulli sampling to convert probability of default into binary labels (default = 1, no default = 0).
This approach yielded a semi-synthetic supervised dataset representing realistic credit default scenarios.

#Objectives
1. Predict the probability of default for rural borrowers.
2. Generate a synthetic “default” variable using logistic and Bernoulli formulations.
3. Preprocess and standardize socioeconomic and financial data.
4. Optimize models using Optuna for maximum performance.
5. Integrate SHAP for model transparency and feature-level interpretability.
6. Deploy the final solution using Streamlit for real-time credit evaluation.

#Methodology
1. Data Preprocessing:
Missing values imputed using median/mode.
Outliers winsorized.
Categorical variables one-hot encoded.
Numerical features z-scaled using StandardScaler.
Engineered financial features:
Loan-to-Income Ratio (LTI)
Income Buffer (IB)
Estimated Savings
Rainfall Deficit (synthetic economic shock)

2. Synthetic Output Generation:
Computed credit risk score (Li) using:
Li = -1.8 + 1.10(Z_LTI) - 0.90(Z_IB) - 0.50(Z_Savings) - 0.30(Z_Tenure) + 0.40(ConsumptionLoan) - 0.20(Agriculture_Loan) + 0.30(Rainfall_Deficit)

Converted Li into a probability of default (Pi) using a sigmoid (logistic) function:
Pi = 1/(1+e^(-Li))

Transformed Pi into binary outputs using a Bernoulli distribution, forming the supervised “default” column.

3. Model Development and Optimization
Trained multiple ML algorithms: Logistic Regression, SVM, Decision Tree, Random Forest, and XGBoost.
Used Optuna for Bayesian optimization based on ROC-AUC metric.
Handled class imbalance with appropriate weighting (scale_pos_weight for XGBoost).

4. Explainability (SHAP)
Used SHAP for both global and local model interpretation.
SHAP visualizations revealed the most influential features in credit risk prediction.

5. Deployment (Streamlit)
Developed a Streamlit app where loan officers can:
Input borrower information.
Get real-time probability of default (PD).
View top features influencing each prediction.

#Key Results
1. ROC-AUC: 97.5%
2. PR-AUC: 94.3%
3. Precision: 83.7%
4. Recall: 86.7%
5. F-1 Score: 85.2%

#Interpretation
High ROC-AUC (97.5%) and PR-AUC (94.3%) show strong discriminative capability.
Balanced precision (83.7%) and recall (86.7%) ensure accurate identification of high-risk borrowers.
F1-score (85.2%) confirms effective tradeoff between false positives and false negatives.

#Explainability Insights
Top contributing features (SHAP importance):
Annual Income, Monthly Income, Estimated Savings
Loan-to-Income Ratio (LTI)
Income Buffer (IB)
SHAP plots confirmed that borrower income and savings stability are the most influential indicators of creditworthiness.

#Streamlit App (Local Deployment)
To run the interactive web app locally:
streamlit run app.py
The app allows:
Manual or file-based borrower data entry
Real-time credit risk prediction
Interactive SHAP-based feature explainability

#Tech Stack
Language: Python 3.10+
Libraries:
pandas, numpy, scikit-learn, xgboost, optuna, shap, matplotlib, seaborn, streamlit
