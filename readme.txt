# Big Market Sales Prediction

This project focuses on predicting the sales of products across different stores in a large retail market using machine learning. The goal is to build a model that can accurately forecast sales based on various product and store features. The dataset used is in CSV format and contains detailed information about products and outlets.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Features](#features)
- [Target](#target)
- [Data Preprocessing](#data-preprocessing)
- [Modeling Techniques](#modeling-techniques)
- [Evaluation](#evaluation)
- [Results](#results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Conclusion](#conclusion)

---

## Project Overview
The **Big Market Sales Prediction** project aims to predict the sales of products in different stores using machine learning. The primary objective is to develop a model that can accurately predict **Item Outlet Sales** based on product and store-related features.

---

## Dataset Description
The dataset contains **8,523 records** and **12 columns**, including product and store attributes. Here’s a breakdown of the columns:

### Features:
1. **Item_Identifier**: Unique ID for each product.
2. **Item_Weight**: Weight of the product.
3. **Item_Fat_Content**: Fat content (e.g., Low Fat, Regular).
4. **Item_Visibility**: Percentage of display area allocated to the product.
5. **Item_Type**: Category of the product (e.g., Dairy, Soft Drinks).
6. **Item_MRP**: Maximum Retail Price of the product.
7. **Outlet_Identifier**: Unique ID for each store.
8. **Outlet_Establishment_Year**: Year the store was established.
9. **Outlet_Size**: Size of the store (e.g., Small, Medium, Large).
10. **Outlet_Location_Type**: Location type of the store (e.g., Tier 1, Tier 2).
11. **Outlet_Type**: Type of store (e.g., Grocery Store, Supermarket).

### Target:
- **Item_Outlet_Sales**: Total sales of the product in a specific store.

---

## Data Preprocessing
1. **Handling Missing Values**:
   - Missing values in `Item_Weight` were filled with the **mean** value.
   - Missing values in `Outlet_Size` were filled with the **mode** based on the `Outlet_Type`.

2. **Label Encoding**:
   - Categorical variables like `Item_Fat_Content`, `Item_Type`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`, and `Outlet_Identifier` were converted into numerical values using label encoding.

3. **Splitting the Dataset**:
   - The dataset was split into **training (80%)** and **testing (20%)** sets using `train_test_split` from Scikit-learn.

---

## Modeling Techniques
The following machine learning algorithm was used:
1. **XGBoost Regressor**: A powerful gradient boosting algorithm known for its performance on structured data.
2. **Evaluation Metric**:
   - **R-squared (R²)**: Measures how well the model explains the variance in the target variable.

---

## Evaluation
1. **Training Data**: The model achieved an **R² score of 0.876** on the training set, indicating a strong fit.
2. **Testing Data**: The R² score on the test set was **0.502**, suggesting some overfitting and room for improvement.

---

## Results
- The **XGBoost Regressor** performed well on the training data with an **R² score of 0.876** but showed moderate performance on the test set with an **R² score of 0.502**.
- While the model captures trends in the training data well, further tuning and feature engineering could improve its generalization on unseen data.

---

## Requirements
To run this project, you need:
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost

Install the required libraries using:
```bash
pip install -r requirements.txt
```

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/big-market-sales-prediction.git
   ```
2. Navigate to the project folder:
   ```bash
   cd big-market-sales-prediction
   ```
3. Open the Jupyter Notebook or Google Colab file and run the code.
4. Ensure the dataset (`Train.csv`) is placed in the same directory as the notebook.

---

## Conclusion
The **Big Market Sales Prediction** project successfully predicts product sales across different stores using machine learning. The **XGBoost Regressor** performed well on the training data but showed room for improvement on the test set. This project highlights the potential of machine learning in solving real-world business problems like sales forecasting. Further tuning and feature engineering could enhance the model's performance.

