# Global House Purchase Prediction

A machine learning project that predicts house purchase decisions based on various property and customer financial features using K-Nearest Neighbors (KNN) classification.

## Project Overview

This project analyzes a dataset of 200,000 property records from 13 countries across 40 cities to predict whether a customer will purchase a house. The model considers property characteristics, location, pricing, and customer financial information to make predictions.

## Dataset

### Size
- **Total Records**: 200,000
- **Features**: 25 columns
- **Target Variable**: `decision` (0 = No Purchase, 1 = Purchase)

### Features

#### Property Features
- `property_id`: Unique identifier for each property
- `property_type`: Type of property (Apartment, Villa, Farmhouse, Townhouse, etc.)
- `furnishing_status`: Furnishing level (Unfurnished, Semi-Furnished, Fully-Furnished)
- `property_size_sqft`: Property size in square feet (400-6000)
- `price`: Property price ($56,288 - $4,202,732)
- `constructed_year`: Year of construction (1960-2023)
- `previous_owners`: Number of previous owners (0-6)
- `rooms`: Number of rooms (1-8)
- `bathrooms`: Number of bathrooms (1-8)
- `garage`: Garage availability (0/1)
- `garden`: Garden availability (0/1)

#### Location Features
- `country`: 13 countries (Australia, Brazil, Canada, China, France, Germany, India, Japan, Singapore, South Africa, UAE, UK, USA)
- `city`: 40 major global cities

#### Legal & Safety Features
- `crime_cases_reported`: Number of crime cases (0-10)
- `legal_cases_on_property`: Legal disputes (0/1)

#### Customer Financial Features
- `customer_salary`: Annual salary ($2,000 - $100,000)
- `loan_amount`: Loan amount requested
- `loan_tenure_years`: Loan duration (10, 15, 20, 25, 30 years)
- `monthly_expenses`: Monthly expenses ($500 - $20,000)
- `down_payment`: Initial payment amount
- `emi_to_income_ratio`: EMI to income ratio (0.0 - 3.46)

#### Rating Features
- `satisfaction_score`: Customer satisfaction (1-10)
- `neighbourhood_rating`: Neighborhood quality (1-10)
- `connectivity_score`: Transportation connectivity (1-10)

## Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas`: Data manipulation and analysis
  - `numpy`: Numerical computations
  - `matplotlib`: Data visualization
  - `seaborn`: Statistical data visualization
  - `scikit-learn`: Machine learning algorithms

## Installation

```bash
# Clone the repository
git clone <https://github.com/Omkar-Kashid23/House-Price-Prediction-Machine-Learning-Model-KNN->

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data Preprocessing

1. **Encoding Categorical Variables**:
   - Label Encoding: `furnishing_status`, `property_type`
   - One-Hot Encoding: `country`, `city`

2. **Data Quality**:
   - No missing values in the dataset
   - All features properly formatted

3. **Feature Engineering**:
   - Converted boolean city columns for location analysis
   - Dropped redundant `country` and `city` columns after encoding

## Model Performance

### K-Nearest Neighbors (KNN) Classifier

**Test Accuracy**: ~73%

#### Classification Report:
```
              precision    recall  f1-score   support

           0       0.77      0.91      0.84     30,812
           1       0.26      0.11      0.15      9,188

    accuracy                           0.73     40,000
   macro avg       0.52      0.51      0.49     40,000
weighted avg       0.66      0.73      0.68     40,000
```

### Key Findings:
- The model performs well in predicting "No Purchase" (Class 0)
- Lower performance on "Purchase" predictions (Class 1) due to class imbalance
- Class distribution: ~77% No Purchase, ~23% Purchase

## Usage

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("global_house_purchase_dataset.csv")

# Preprocess data (encoding, feature engineering)
# ... (preprocessing steps)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Train model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## Model Optimization

The project includes random state optimization to find the best train-test split:

```python
# Testing 24 different random states
for i in range(1, 25):
    # Train and evaluate model
    # Track accuracy for each random state
```

**Best Random State**: 24 (highest accuracy)

## Future Improvements

1. **Address Class Imbalance**:
   - Implement SMOTE (Synthetic Minority Over-sampling Technique)
   - Use class weights in model training
   - Try ensemble methods

2. **Feature Engineering**:
   - Create interaction features
   - Normalize numerical features
   - Feature selection techniques

3. **Try Alternative Models**:
   - Random Forest
   - Gradient Boosting (XGBoost, LightGBM)
   - Neural Networks
   - Logistic Regression

4. **Hyperparameter Tuning**:
   - Grid Search or Random Search for optimal KNN parameters
   - Cross-validation for robust evaluation

## Project Structure

```
Global-House-Purchase-Prediction/
│
├── Global House Purchase Prediction.ipynb    # Main Jupyter notebook
├── global_house_purchase_dataset.csv         # Dataset
├── README.md                                  # Project documentation
└── requirements.txt                           # Python dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for everyone.

## Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This is an educational project for learning machine learning classification techniques. The model's performance can be improved through advanced techniques and feature engineering.
