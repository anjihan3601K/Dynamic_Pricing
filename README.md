# 🚀 Dynamic Pricing Model for Ride-Sharing

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24%2B-orange.svg)](https://scikit-learn.org/stable/)
[![Pandas](https://img.shields.io/badge/Pandas-lightgreen.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-red.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-purple.svg)](https://seaborn.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-blueviolet.svg)](https://plotly.com/python/)

This repository contains the code and analysis for a dynamic pricing model, which aims to predict the optimal ride cost in a ride-sharing service to balance supply and demand. The project demonstrates a full data science workflow, from exploratory data analysis (EDA) to advanced model building and evaluation.

---

## 🎯 1. Problem Statement

The core objective of this project is to develop a robust machine learning model capable of **dynamically adjusting ride prices**. This adjustment is based on real-time market conditions and aims to **optimize profitability** for the ride-sharing service while simultaneously ensuring a **reliable and efficient service** for customers.

---

## 📊 2. Data

The analysis is built upon the `dynamic_pricing.csv` dataset, which comprises **1,000 entries** and **10 distinct columns** [cite: uploaded:model_DP.ipynb]. This rich dataset provides a comprehensive view of various factors influencing ride-sharing transactions.

#### Key Features:
* `Number_of_Riders`: The current demand for rides in a given area.
* `Number_of_Drivers`: The available supply of drivers in that area.
* `Location_Category`: Geographical classification (e.g., Urban, Suburban, Rural).
* `Customer_Loyalty_Status`: Customer segmentation based on loyalty.
* `Number_of_Past_Rides`: Historical usage data for each customer.
* `Average_Ratings`: Service quality feedback.
* `Time_of_Booking`: Temporal aspect of the ride request (e.g., Morning, Afternoon, Evening, Night).
* `Vehicle_Type`: The class of vehicle requested (e.g., Economy, Premium).
* `Expected_Ride_Duration`: The estimated length of the trip.
* `Historical_Cost_of_Ride`: The baseline cost before any dynamic adjustments.

---

## 📈 3. Exploratory Data Analysis (EDA)

The `EDA-Dynamic pricing.ipynb` notebook meticulously explores the dataset to uncover patterns and relationships. Key findings from the EDA include:

* **Direct Supply-Demand Relationship**: A clear and impactful correlation was observed between the **ratio of riders to drivers** and the `Adjusted_Ride_Cost`. This highlights the direct influence of market dynamics on pricing [cite: uploaded:model_DP.ipynb].
* **Enhanced Profitability**: Visualizations confirm that the implemented dynamic pricing strategy significantly increased the percentage of **profitable rides** compared to the historical pricing model, validating the business impact [cite: uploaded:model_DP.ipynb].

---

## 🛠️ 4. Methodology & Model Building

The `model_DP.ipynb` notebook outlines the end-to-end machine learning pipeline, from raw data to a predictive model.

#### **Preprocessing Pipeline**
A robust scikit-learn `Pipeline` was established to ensure consistent data transformations and prevent data leakage, a critical step for model generalization [cite: uploaded:model_DP.ipynb]. This pipeline incorporates a `ColumnTransformer` for:
* **Feature Engineering**: A crucial `driver_to_rider_ratio` feature was engineered to directly quantify supply-demand imbalances, a core driver for dynamic pricing [cite: uploaded:model_DP.ipynb].
* **Numerical Scaling**: `StandardScaler` was applied to all numerical features, normalizing their scale to prevent dominance by features with larger values [cite: uploaded:model_DP.ipynb].
* **Categorical Encoding**: `OneHotEncoder` converted all nominal categorical features into a numerical format, making them suitable for machine learning algorithms [cite: uploaded:model_DP.ipynb].

#### **Model Selection and Tuning**
A comparative analysis of several regression models was performed to identify the most suitable algorithm for this dataset [cite: uploaded:model_DP.ipynb]. The **`GradientBoostingRegressor`** consistently demonstrated superior performance and was selected as the base model for further optimization [cite: uploaded:model_DP.ipynb].

* **Hyperparameter Tuning**: `GridSearchCV` was extensively used to systematically search for the optimal combination of hyperparameters for the `GradientBoostingRegressor`, ensuring peak performance on the training data [cite: uploaded:model_DP.ipynb].

#### **Advanced Technique: Stacking Ensemble**
To further boost predictive accuracy and model robustness, an advanced **`StackingRegressor`** was implemented. This ensemble method intelligently combined the predictions from a `RandomForestRegressor` and a `GradientBoostingRegressor`, leveraging their individual strengths [cite: uploaded:model_DP.ipynb]. This approach resulted in a lower RMSE compared to the single best model, showcasing the power of ensemble learning [cite: uploaded:model_DP.ipynb].

---

## 📈 5. Results & Key Metrics

The final model's performance was rigorously evaluated on a dedicated, unseen test set using industry-standard regression metrics:

* **R-squared ($R^2$)**: **0.8434** - Indicating that approximately 84.34% of the variance in ride cost can be explained by the model's features [cite: uploaded:model_DP.ipynb].
* **Mean Absolute Error (MAE)**: **64.4904** - Meaning, on average, the model's predictions are off by approximately $64.49, providing a clear, interpretable measure of error [cite: uploaded:model_DP.ipynb].
* **Root Mean Squared Error (RMSE)**: **86.6581** - Penalizes larger errors more heavily, giving insight into the magnitude of typical prediction errors [cite: uploaded:model_DP.ipynb].

These metrics collectively confirm the model's strong predictive capabilities, making it a valuable tool for dynamic pricing optimization.

---

## 🏃‍♀️ 6. How to Run the Code

To explore and replicate this dynamic pricing model:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/dynamic-pricing-model.git](https://github.com/your-username/dynamic-pricing-model.git)
    cd dynamic-pricing-model
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt # (assuming you create this file)
    # or manually install:
    # pip install pandas numpy matplotlib seaborn scikit-learn plotly
    ```
3.  **Open Jupyter Notebooks:**
    ```bash
    jupyter notebook
    ```
    Then navigate to:
    * `dynamic_pricing.csv`: The raw dataset.
    * `EDA-Dynamic pricing.ipynb`: For in-depth exploratory data analysis.
    * `model_DP.ipynb`: For the complete machine learning pipeline, including preprocessing, model training, and evaluation.

---

**Tags:** `machine-learning`, `data-science`, `dynamic-pricing`, `regression`, `python`, `scikit-learn`, `eda`, `feature-engineering`, `ensemble-learning`, `stacking`, `gradient-boosting`, `random-forest`
