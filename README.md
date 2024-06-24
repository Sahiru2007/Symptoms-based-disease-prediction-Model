

# Symptom-Based Disease Prediction Model

This repository contains a Jupyter Notebook for building and evaluating a machine learning model to predict diseases based on symptoms. The model uses multiple classifiers and combines their predictions to provide a final diagnosis.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Saving the Model](#saving-the-model)
- [Contributing](#contributing)
- [License](#license)

## Overview

The notebook builds and evaluates several machine learning models to predict diseases based on symptoms. The steps include:

1. Loading and exploring the dataset.
2. Data preprocessing.
3. Splitting the data into training and testing sets.
4. Building multiple classifiers.
5. Combining predictions from different models.
6. Evaluating the models using various metrics.
7. Saving the best model.

## Dataset

The dataset used in this notebook consists of symptoms and corresponding diseases. Each row represents a patient with a set of symptoms and the diagnosed disease. The dataset can be found [here](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset).

### Dataset Description

- **Symptom1**: Presence (1) or absence (0) of Symptom 1
- **Symptom2**: Presence (1) or absence (0) of Symptom 2
- **...**
- **SymptomN**: Presence (1) or absence (0) of Symptom N
- **Disease**: The diagnosed disease

## Installation

To run this notebook, you need to have Python installed along with the following packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these packages using pip:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn
```

Alternatively, you can use the provided `requirements.txt` file:

```sh
pip install -r requirements.txt
```

## Usage

1. Clone this repository:

```sh
git clone [<repository-url>](https://github.com/Sahiru2007/Symptoms-based-disease-prediction-Model.git)
cd Symptoms-based-disease-prediction-Model
```

2. Open the Jupyter Notebook:

```sh
jupyter notebook Symptom_based_disease_prediction.ipynb
```

3. Run all cells in the notebook to see the complete analysis and model evaluation.

## Data Preprocessing

### Handling Missing Values

Missing values in the dataset are handled by replacing them with zeros, as the dataset represents the presence or absence of symptoms.

### Encoding Labels

The disease labels are encoded into numeric values using label encoding.

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data['Disease'] = encoder.fit_transform(data['Disease'])
```

### Splitting the Data

The dataset is split into training and testing sets using a 70-30 split:

```python
from sklearn.model_selection import train_test_split

X = data.drop('Disease', axis=1)
y = data['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## Model Building

### Models Used

The notebook evaluates several models:

- **Random Forest**
- **Naive Bayes**
- **Support Vector Machine (SVM)**

### Training the Models

Example: Training a Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
```

## Model Evaluation

### Metrics

The models are evaluated using the following metrics:

- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Precision**: The ratio of true positive instances to the total predicted positives.
- **Recall**: The ratio of true positive instances to the actual positives.
- **F1 Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A summary of prediction results on a classification problem.

### Example: Evaluating Random Forest Classifier

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_test, rf_predictions)
precision = precision_score(y_test, rf_predictions, average='weighted')
recall = recall_score(y_test, rf_predictions, average='weighted')
f1 = f1_score(y_test, rf_predictions, average='weighted')
cm = confusion_matrix(y_test, rf_predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix: \n{cm}')
```

### Evaluation Results

- **Random Forest**: Accuracy ~ 85%
- **Naive Bayes**: Accuracy ~ 80%
- **SVM**: Accuracy ~ 82%

## Saving the Model

The best-performing model is saved using the `pickle` module for future use:

```python
import pickle

filename = 'disease_prediction_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(rf_model, file)

print(f"Model saved to {filename}")
```

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---
