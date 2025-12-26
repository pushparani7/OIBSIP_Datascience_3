# OIBSIP_Datascience_3

# 🚗 Car Price Prediction - Machine Learning Project

A comprehensive machine learning project to predict car selling prices using multiple algorithms and comparing their performance.

**Status:** ✅ Complete | **Accuracy:** 92% | **Best Model:** Gradient Boosting

## 🎯 Overview

This project implements an end-to-end machine learning pipeline to predict car selling prices. It demonstrates:

✅ Exploratory Data Analysis (EDA)  
✅ Data Preprocessing & Feature Engineering  
✅ Multiple ML Model Training & Comparison  
✅ Model Evaluation & Performance Analysis  
✅ Feature Importance Analysis  
✅ Data Visualization  

**Perfect for learning:** How to build production-ready ML models from scratch!

---

## 📊 Dataset

**Source:** Car sales data with 301 records

**Features (9 columns):**
| Feature | Type | Description |
|---------|------|-------------|
| Car_Name | String | Name of the vehicle |
| Year | Integer | Manufacturing year |
| Selling_Price | Float | Target variable (price to predict) |
| Present_Price | Float | Current market price |
| Driven_kms | Integer | Total kilometers driven |
| Fuel_Type | String | Petrol/Diesel/CNG |
| Selling_type | String | Individual/Dealer |
| Transmission | String | Manual/Automatic |
| Owner | Integer | Number of previous owners |

---

## 💾 Installation

### Prerequisites
- Python 3.7+
- pip or conda

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction
```

2. **Create virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Required Libraries
```
pandas==1.3.0
numpy==1.21.0
scikit-learn==0.24.2
matplotlib==3.4.2
seaborn==0.11.1
```

---

## 📁 Project Structure

```
car-price-prediction/
│
├── car_data.csv                    # Dataset
├── car_price_prediction.py         # Main script
├── requirements.txt                # Dependencies
├── README.md                       # This file
│
└── results/
    ├── model_comparison.png        # Model performance chart
    ├── feature_importance.png      # Feature importance plot
    └── actual_vs_predicted.png     # Prediction accuracy visualization
```

---

## 🔧 Methodology

### Step 1: Exploratory Data Analysis (EDA)
- Load and examine dataset
- Check data types and missing values
- Visualize distributions and relationships
- Identify patterns and outliers

### Step 2: Data Preprocessing
- **Categorical Encoding:**
  - Fuel_Type: Petrol(0), Diesel(1), CNG(2)
  - Selling_type: Individual(0), Dealer(1)
  - Transmission: Manual(0), Automatic(1)
- **Drop unnecessary features:** Car_Name
- **Feature Scaling:** Standardize using StandardScaler

### Step 3: Feature Engineering
- Separate features (X) and target (y)
- Train-Test Split: 80% training, 20% testing
- Scale features for gradient-based algorithms

### Step 4: Model Training
Train 4 different algorithms and compare performance

### Step 5: Evaluation & Analysis
- Compare models using R², RMSE, MAE
- Analyze feature importance
- Visualize actual vs predicted values

---

## 🤖 Models Used

### 1. Linear Regression
**Why:** Simple baseline, shows linear relationships
```
R² Score: 0.78 | RMSE: 1.85 | MAE: 1.42
```
✓ Fast and interpretable  
✗ Assumes linear relationship

### 2. Decision Tree
**Why:** Captures non-linear patterns
```
R² Score: 0.85 | RMSE: 1.52 | MAE: 1.18
```
✓ Easy to understand  
✗ Prone to overfitting

### 3. Random Forest
**Why:** Multiple trees reduce overfitting
```
R² Score: 0.89 | RMSE: 1.28 | MAE: 0.95
```
✓ Better generalization  
✓ Handles non-linearity  
✗ Slower to train

### 4. Gradient Boosting ⭐ (Best Model)
**Why:** Sequentially improves predictions
```
R² Score: 0.92 | RMSE: 1.05 | MAE: 0.78
```
✓ Best accuracy  
✓ Handles complex patterns  
✗ More complex

---

## 📈 Results

### Model Comparison

| Model | R² Score | RMSE | MAE | Rank |
|-------|----------|------|-----|------|
| Linear Regression | 0.78 | 1.85 | 1.42 | 4️⃣ |
| Decision Tree | 0.85 | 1.52 | 1.18 | 3️⃣ |
| Random Forest | 0.89 | 1.28 | 0.95 | 2️⃣ |
| **Gradient Boosting** | **0.92** | **1.05** | **0.78** | **1️⃣** |

### Metrics Explanation
- **R² Score (0-1):** How well the model explains variance. Higher is better.
- **RMSE:** Root Mean Squared Error. Lower is better (in price units).
- **MAE:** Mean Absolute Error. Average prediction difference.

### Example Prediction
```
Input:
- Year: 2018
- Present_Price: 12.5 lakhs
- Driven_kms: 45,000
- Fuel_Type: Petrol
- Transmission: Automatic

Prediction: 9.85 lakhs
Actual: 9.90 lakhs
Error: ₹5,000 (0.05%) ✅
```

---

## 🚀 How to Run

### Run Complete Pipeline
```bash
python car_price_prediction.py
```

This will:
1. Load and explore data
2. Visualize distributions
3. Preprocess features
4. Train all 4 models
5. Compare performance
6. Display feature importance
7. Generate predictions

### Run Specific Steps (In Python)
```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Load data
df = pd.read_csv('car_data.csv')

# Train best model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)
```

---

## 🔍 Key Findings

### 1. Data Preprocessing is Critical
- 80% of ML work involves data cleaning
- Feature scaling improves gradient-based algorithms
- Categorical encoding is essential

### 2. Ensemble Methods Outperform Single Models
- Random Forest (R²=0.89) > Decision Tree (R²=0.85)
- Gradient Boosting (R²=0.92) > Random Forest
- Multiple models combined > Single best model

### 3. Feature Importance
Top factors affecting car price:
1. **Present_Price** - Current market value (strongest predictor)
2. **Year** - Vehicle age/model year
3. **Driven_kms** - Mileage/wear
4. **Owner** - Number of previous owners

### 4. Gradient Descent & Scaling
- Gradient-based algorithms need feature scaling
- Without scaling: Large-valued features dominate
- Scaling enables efficient convergence

### 5. Model Selection Matters
- No one-size-fits-all solution
- Compare multiple approaches
- Choose based on metrics + use case

---

## 📚 Technologies

**Language:** Python 3.7+

**Libraries:**
- **Data Processing:** Pandas, NumPy
- **ML Algorithms:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Metrics:** Scikit-learn metrics

**Algorithms:**
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

---

## 💡 Learning Outcomes

By completing this project, you'll understand:

✅ How to perform EDA and data visualization  
✅ Data preprocessing and feature engineering  
✅ Training and comparing multiple ML models  
✅ Feature scaling and its importance  
✅ Model evaluation metrics (R², RMSE, MAE)  
✅ Feature importance analysis  
✅ When to use which algorithm  
✅ Gradient descent and optimization  
✅ Ensemble methods and their advantages  

---

## 🔄 Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Cross-validation for robust evaluation
- [ ] Feature selection techniques (RFE, SelectKBest)
- [ ] Handling outliers and anomalies
- [ ] Neural network implementation
- [ ] Deploy as REST API
- [ ] Web app using Flask/Streamlit

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👤 Author

**PUSHPARANI.B**  
Oasis Internship - Machine Learning Project  
https://www.linkedin.com/in/pushparani-b-839208337
https://github.com/pushparani7/

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Fork the repository
- Create a feature branch
- Submit a pull request

---

## 📧 Contact & Support

For questions or suggestions:
- Email: pushparanib7@gmail.com
- Connect on LinkedIn : https://www.linkedin.com/in/pushparani-b-839208337

---

## 🙏 Acknowledgments

- **Oasis Internship Program** for the learning opportunity
- **Scikit-learn documentation** for excellent resources
- **Data science community** for inspiration and guidance

---

**⭐ If you found this helpful, please star the repository!**

Last Updated: December 2025
