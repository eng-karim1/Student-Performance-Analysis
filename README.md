# Student-Performance-Analysis
## 📌 Project Overview
This project analyzes various factors affecting student performance, using statistical and machine learning techniques. The dataset contains information on study habits, motivation levels, and exam scores. The goal is to identify key factors influencing students' academic success and build predictive models.

## 📊 Data Exploration
- **Dataset**: `StudentPerformanceFactors.csv`
- **Key Features**:
  - `Hours_Studied`: Study hours per week
  - `Motivation_Level`: Self-reported motivation level
  - `Exam_Score`: Final exam score
  
### Exploratory Data Analysis (EDA)
- Statistical summary and data distribution visualization.
- Identification of missing values and outliers.
- Correlation analysis to find relationships between variables.

## 🛠 Data Preprocessing
- Handling missing values by replacing them with the mean.
- Encoding categorical variables.
- Scaling numerical data using `StandardScaler`.

## 🤖 Machine Learning Models
### Models Implemented:
1. **Linear Regression**: For predicting exam scores based on study habits.
2. **Random Forest Regressor**: A more robust model for handling nonlinear relationships.
3. **Gradient Boosting & XGBoost**: Advanced models for better predictive accuracy.

### Model Evaluation Metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score** (Goodness of fit)

## 🔍 Key Findings
- Strong correlation found between `Hours_Studied` and `Exam_Score`.
- Motivation level also plays a significant role in student performance.
- **Linear Regression provided the best predictive accuracy** in this analysis.
- **Random Forest performed worse than expected**, indicating it may not be the best choice for this dataset.

## 📁 Project Structure
```
├── StudentPerformanceFactors.csv  # Dataset
├── student_performance_analysis.py  # Main script
├── README.md  # Project documentation
├── requirements.txt  # Required Python packages
└── notebook.ipynb  # Jupyter Notebook (optional)
```

## 🚀 How to Use
### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 2️⃣ Run the Analysis
```bash
python student_performance_analysis.py
```

## 📌 Future Improvements
- Add more features such as sleep patterns and extracurricular activities.
- Implement deep learning models for enhanced predictions.
- Optimize hyperparameters of Random Forest to improve performance.

## 📜 License
This project is open-source and available for educational purposes.

---
💡 **Contributions and feedback are welcome!** Feel free to fork the repo and improve the analysis. 🚀
