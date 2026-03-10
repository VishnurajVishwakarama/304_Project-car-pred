# \ud83d\ude97 Car Price Prediction Project

A complete Machine Learning workflow and interactive Web Application that estimates the market price of used cars based on their technical specifications. 

This project covers end-to-end Data Science procedures, including Data Cleaning, Exploratory Data Analysis (EDA), Feature Engineering, Model Training, and a deployment-ready Streamlit interface.

---

## \ud83d\udcc2 Project Structure

```bash
.
\u251c\u2500\u2500 cars_price_pred.ipynb       # Main Jupyter Notebook containing EDA, Feature Engineering & ML Models
\u2514\u2500\u2500 car_price_streamlit_app/    # Web Application Deployment Folder
    \u251c\u2500\u2500 app.py                  # Streamlit Frontend UI & Backend Logic
    \u251c\u2500\u2500 car_price_model.pkl     # Pre-trained XGBoost / RandomForest Model
    \u2514\u2500\u2500 requirements.txt        # Python dependencies for the web app
```

---

## \ud83d\udcca 1. Exploring the Jupyter Notebook

The notebook (`cars_price_pred.ipynb`) is the heart of the analytical workflow. It processes a raw dataset of 205 cars and determines which specifications have the greatest impact on pricing.

### What's Inside?
- **Data Cleaning**: Handling missing values, discarding duplicate records, and addressing outliers.
- **Exploratory Data Analysis**: Univariate, Bivariate, and Multivariate visualizations to understand variable distributions and correlations.
- **Feature Engineering**: Creating interaction features (like `car_volume`), extracting polynomial combinations, and encoding categorical variables using `LabelEncoder`.
- **Feature Selection**: Dropping heavily correlated columns (preventing multicollinearity) and utilizing `SelectKBest`, `RFE`, and `RandomForest` importance rankings to distill the dataset down to the most influential predictors.
- **Model Training & Evaluation**: Training robust non-linear models including `DecisionTree`, `RandomForest`, `GradientBoosting`, and `XGBoost`. Performance is scored on an 80-20 train-test split utilizing $R^2$, `RMSE`, and `MAE`.

### How to use the Notebook
You can open this notebook using Jupyter natively or directly inside VS Code:
1. Ensure your `.venv` is activated.
2. Select the `.venv` as your active IPython kernel in the top right corner.
3. Click **Run All** to execute the pipeline from start to finish.

---

## \ud83c\udf10 2. Running the Streamlit Web Application

We extracted the highest-performing trained model from our notebook (`car_price_model.pkl`) and linked it to a dynamic web interface using Streamlit. This allows any user to input custom car features and instantly receive a predicted market price.

### Installation & Setup

1. **Activate your Virtual Environment** (If not already active):
   ```bash
   source .venv/bin/activate
   ```

2. **Navigate to the App Directory**:
   ```bash
   cd car_price_streamlit_app
   ```

3. **Install Dependencies**:
   Install all necessary packages via the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Server**:
   Execute the Streamlit runtime:
   ```bash
   streamlit run app.py
   ```

5. **Interact**: 
   Your default web browser should automatically open a new tab pointed to `http://localhost:8501`. If it doesn't, navigate to that link manually. You can now tweak the sliders for Engine Size, Horsepower, Car Width, Curb Weight, and MPG to see real-time price predictions!

---

## \u2699\ufe0f Technologies Used
- **Language**: Python 3.14+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost
- **Data Visualization**: Matplotlib, Seaborn
- **Web Deployment**: Streamlit
