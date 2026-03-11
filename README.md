# 🚗 Car Price Prediction — End-to-End Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-ScikitLearn-orange)
![Model](https://img.shields.io/badge/Model-XGBoost-green)
![Deployment](https://img.shields.io/badge/WebApp-Streamlit-red)
![Status](https://img.shields.io/badge/Project-Complete-success)

An **end-to-end Machine Learning project** that predicts the **market price of used cars** based on vehicle specifications.

This project demonstrates the **complete Data Science workflow**, including:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Selection
- Model Training
- Model Evaluation
- Model Deployment using Streamlit

The final result is an **interactive web application** where users can input car specifications and instantly receive a predicted price.

---

# 📊 Project Architecture

```
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Feature Engineering
     │
     ▼
Feature Selection
     │
     ▼
Model Training
     │
     ▼
Model Evaluation
     │
     ▼
Best Model Selection
     │
     ▼
Model Serialization (.pkl)
     │
     ▼
Streamlit Web Application
```

---

# 📂 Project Structure

```bash
.
├── cars_price_pred.ipynb
│   └── Full ML workflow (EDA + Feature Engineering + Model Training)
│
└── car_price_streamlit_app
    ├── app.py
    │   └── Streamlit Web Application
    │
    ├── car_price_model.pkl
    │   └── Serialized trained model
    │
    └── requirements.txt
        └── Python dependencies
```

---

# 📊 Dataset Description

The dataset contains **205 automobile records** with technical specifications.

## Target Variable

```
price
```

## Example Features

| Feature | Description |
|------|------|
| CarName | Brand and model of the car |
| engine-size | Engine displacement |
| horsepower | Engine power |
| curb-weight | Weight of the vehicle |
| car-width | Width of the car |
| highway-mpg | Fuel efficiency |
| price | Market price of the car |

The goal is to **predict the price of a car based on its specifications**.

---

# 🔬 Machine Learning Development Lifecycle (MLDLC)

## 1️⃣ Problem Understanding

**Business Problem**

Estimate the **market price of a car** using its technical specifications.

**Machine Learning Problem Type**

```
Supervised Regression
```

---

# 🧹 2️⃣ Data Cleaning

Performed several preprocessing steps:

- Handling missing values
- Removing duplicate rows
- Fixing inconsistent brand names
- Removing unnecessary columns
- Detecting outliers

Example:

```python
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
```

---

# 📊 3️⃣ Exploratory Data Analysis (EDA)

EDA helps understand **patterns, distributions, and relationships** in the dataset.

## Univariate Analysis

Distribution of numerical variables.

```python
sns.histplot(df["price"], kde=True)
```

## Bivariate Analysis

Relationship between features and price.

```python
sns.scatterplot(x=df["engine-size"], y=df["price"])
```

## Correlation Analysis

```python
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
```

### Key Insights

Important features affecting price:

- Engine Size
- Horsepower
- Curb Weight
- Car Width

---

# ⚙️ 4️⃣ Feature Engineering

Feature engineering improves model performance by creating meaningful features.

## Extract Car Brand

```python
df["CarBrand"] = df["CarName"].apply(lambda x: x.split(" ")[0])
```

## Create Interaction Feature

```python
df["car_volume"] = df["carlength"] * df["carwidth"] * df["carheight"]
```

## Encode Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["fueltype"] = encoder.fit_transform(df["fueltype"])
```

---

# 🎯 5️⃣ Feature Selection

Feature selection helps remove irrelevant features and reduce overfitting.

Methods used:

- Correlation filtering
- SelectKBest
- Recursive Feature Elimination (RFE)
- RandomForest Feature Importance

Example:

```python
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X, y)
```

---

# 🤖 6️⃣ Model Training

Several regression models were trained and compared.

Models used:

- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Example:

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train, y_train)
```

---

# 📈 7️⃣ Model Evaluation

Model performance was evaluated using:

| Metric | Description |
|------|------|
| R² Score | Variance explained by the model |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |

Example:

```python
from sklearn.metrics import r2_score, mean_absolute_error

pred = model.predict(X_test)

r2_score(y_test, pred)
mean_absolute_error(y_test, pred)
```

---

# 🏆 Model Comparison

| Model | Performance |
|------|------|
| Decision Tree | Baseline |
| Random Forest | Improved |
| Gradient Boosting | Strong |
| XGBoost | **Best Performance** |

The **best-performing model was saved for deployment**.

---

# 💾 Model Serialization

The trained model was saved using **Pickle**.

```python
import pickle

pickle.dump(model, open("car_price_model.pkl", "wb"))
```

This file is used in the **Streamlit application**.

---

# 🌐 Streamlit Web Application

A **Streamlit-based web interface** allows users to interact with the model.

Users can adjust parameters like:

- Engine Size
- Horsepower
- Car Width
- Curb Weight
- Highway MPG

The model instantly predicts the **estimated car price**.

---

# ▶️ Running the Project

## 1️⃣ Activate Virtual Environment

```bash
source .venv/bin/activate
```

## 2️⃣ Navigate to Streamlit App

```bash
cd car_price_streamlit_app
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 4️⃣ Run Streamlit

```bash
streamlit run app.py
```

---

# 🌍 Open the Web Application

Visit in your browser:

```
http://localhost:8501
```

---

# ⚙️ Technologies Used

| Category | Tools |
|------|------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn, XGBoost |
| Deployment | Streamlit |

---

# 🎯 Skills Demonstrated

This project demonstrates:

- End-to-End Machine Learning Pipeline
- Data Cleaning and Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- Model Training and Evaluation
- Model Deployment using Streamlit

---

# 🚀 Future Improvements

Possible improvements:

- Deploy on **Streamlit Cloud**
- Add **Hyperparameter Tuning**
- Implement **Cross Validation**
- Add **SHAP Explainability**
- Build **REST API for predictions**

---

⭐ If you found this project useful, consider **starring the repository**.