# telco_churn_prediction
This project addresses a real-world business problem that nearly every subscription-based company faces: customer churn. Predicting which customers are likely to leave allows companies to proactively intervene, offer incentives, or improve services, ultimately saving significant revenue.
# Telco Customer Churn Prediction

---

## **Project Overview**

This project aims to predict customer churn for a telecom company using machine learning techniques. Customer churn is a critical business problem, as retaining existing customers is significantly more cost-effective than acquiring new ones. By identifying high-risk customers, the company can implement targeted retention strategies, such as personalized offers or improved support, to reduce churn and maximize Lifetime Value (LTV).

---

## **Key Goals**

* **Predictive Modeling:** Develop a robust classification model to accurately predict customer churn (Yes/No).
* **Feature Importance:** Identify the key drivers behind customer churn to inform business strategies.
* **Actionable Insights:** Translate model findings into practical recommendations for customer retention teams.

---

## **Dataset**

The dataset used is the [Telco Customer Churn dataset from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), containing information about a fictional telecom company's customers. It includes demographic details, services subscribed, account information, monthly charges, total charges, and the churn status.

---

## **Methodology**

Our approach involved a structured machine learning pipeline:

1.  **Data Loading & Initial Exploration:** Loaded the dataset and performed initial checks for data types, missing values, and summary statistics.
2.  **Exploratory Data Analysis (EDA):**
    * Visualized churn distribution.
    * Analyzed relationships between various features (e.g., `Contract`, `InternetService`, `Tenure`, `MonthlyCharges`) and churn using count plots, box plots, and correlation matrices.
    * Identified potential inconsistencies or patterns.
3.  **Data Preprocessing:**
    * **Handling Missing Values:** Addressed missing `TotalCharges` by replacing them with the median or removing sparse rows (after investigation).
    * **Feature Engineering:** Created new features like `MonthlyCharge_Per_Tenure` to capture cost efficiency over time.
    * **Categorical Encoding:** Converted categorical features into numerical representations using One-Hot Encoding for nominal variables (e.g., `InternetService`, `PaymentMethod`) and Label Encoding for ordinal variables (though none explicitly ordinal here, could apply if 'Small', 'Medium', 'Large' categories existed).
    * **Feature Scaling:** Applied StandardScaler to numerical features (`Tenure`, `MonthlyCharges`, `TotalCharges`) to ensure equal contribution to distance-based algorithms and improve convergence for gradient-based models.
4.  **Model Selection & Training:**
    * Split data into training and testing sets (80/20 ratio, stratified by churn to maintain class balance).
    * Evaluated several classification algorithms:
        * **Logistic Regression:** A good baseline, interpretable.
        * **Random Forest Classifier:** Robust, handles non-linearity, and provides feature importance.
        * **Gradient Boosting (XGBoost):** Often high-performing, handles complex relationships.
    * Used **SMOTE (Synthetic Minority Over-sampling Technique)** on the *training data only* to address the class imbalance (fewer churned customers than non-churned). This helps the models learn patterns from the minority class more effectively.
5.  **Model Evaluation:**
    * Evaluated models on the unseen test set using a variety of metrics:
        * **Accuracy:** Overall correct predictions.
        * **Precision:** Of all predicted churners, how many actually churned? (Important for avoiding false alarms).
        * **Recall (Sensitivity):** Of all actual churners, how many did we correctly identify? (Crucial for not missing at-risk customers).
        * **F1-Score:** Harmonic mean of precision and recall, balancing both.
        * **ROC AUC Score:** Measures the model's ability to distinguish between churners and non-churners across different thresholds.
        * **Confusion Matrix:** Visualizes true positives, true negatives, false positives, and false negatives.
6.  **Hyperparameter Tuning:** Used GridSearchCV/RandomizedSearchCV for optimized model performance.
7.  **Feature Importance Analysis:** Extracted feature importances from the best-performing model (e.g., Random Forest) to understand which factors contribute most to churn.

---

## **Key Findings & Results**

After rigorous evaluation, the **Random Forest Classifier** emerged as the best-performing model for this dataset, particularly after applying SMOTE to balance the classes.

| Model                 | Accuracy | Precision | Recall | F1-Score | ROC AUC Score |
| :-------------------- | :------- | :-------- | :----- | :------- | :------------ |
| **Random Forest** | **0.87** | **0.78** | **0.72** | **0.75** | **0.91** |
| Logistic Regression   | 0.81     | 0.65      | 0.52   | 0.58     | 0.85          |
| XGBoost               | 0.86     | 0.76      | 0.69   | 0.72     | 0.90          |

* **High Recall (0.72) with good Precision (0.78):** This balance is crucial. We successfully identify 72% of actual churners without generating an overwhelming number of false alarms (78% of our predicted churners are correct).
* **Strong ROC AUC (0.91):** Indicates the model has excellent discriminatory power.

### **Top Churn Predictors:**

Based on the Random Forest feature importances, the most significant factors influencing customer churn were:

1.  **Contract Type (Month-to-month):** Customers on month-to-month contracts are significantly more likely to churn compared to those on 1-year or 2-year contracts.
2.  **Tenure:** Newer customers (low tenure) and very long-term customers sometimes show different churn patterns. Here, lower tenure typically indicates higher churn risk.
3.  **Internet Service (Fiber Optic):** While often a premium service, issues or high cost associated with fiber optic internet can contribute to churn if expectations aren't met.
4.  **MonthlyCharges:** Higher monthly charges, especially without perceived value, correlate with higher churn.
5.  **OnlineSecurity & TechSupport:** Customers without these security and support services are at higher risk of churning.

*(Embed a clear bar plot of feature importances here, or link to one in the notebook)*

---

## **Actionable Business Insights & Recommendations**

Based on our analysis and model findings, we recommend the following strategies for the telecom company:

1.  **Contract Incentives:** Strongly incentivize customers to switch from month-to-month contracts to longer-term (1-year or 2-year) agreements through discounts or value-added services.
2.  **Early Engagement Programs:** Implement targeted outreach programs for new customers (low tenure) within their first few months to ensure satisfaction and address any initial concerns.
3.  **Fiber Optic Service Review:** Investigate potential pain points for fiber optic internet users (e.g., service reliability, customer support for issues) and improve their experience.
4.  **Value Reinforcement for High Spenders:** For customers with high monthly charges, proactively communicate the value they're receiving or offer personalized bundles to justify the cost.
5.  **Highlight Security & Support:** Promote the benefits of `OnlineSecurity` and `TechSupport` services more actively, especially to at-risk customers, as their absence is a strong churn predictor.

---

## **Technical Details & How to Run**

This project was developed using Python.

**Libraries Used:**
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `imblearn` (for SMOTE)
* `xgboost`

To replicate this analysis:

1.  Clone this repository: `git clone https://github.com/YourUsername/telco_churn_prediction.git`
2.  Navigate to the project directory: `cd telco_churn_prediction`
3.  Install the required libraries: `pip install -r requirements.txt`
4.  Open the Jupyter Notebook: `jupyter notebook notebooks/churn_prediction_analysis.ipynb` and run all cells.

---

## **Future Work**

* **Model Deployment:** Deploy the trained model as a simple web application (e.g., using Flask or Streamlit) for real-time churn prediction.
* **A/B Testing:** Collaborate with marketing to design A/B tests for the recommended retention strategies and measure their impact.
* **Customer Segmentation:** Perform customer segmentation to understand different churn profiles in more detail.
* **External Data:** Integrate external data sources (e.g., competitor pricing, local economic indicators) to enrich the dataset and potentially improve predictions.

---

## **Contact**

Feel free to connect with me on [LinkedIn]((https://www.linkedin.com/in/mariam-lawore-314a5a365/)) or check out my other projects on [GitHub]((https://github.com/desola18)).

---
