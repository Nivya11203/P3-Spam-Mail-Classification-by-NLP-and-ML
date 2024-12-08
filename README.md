# P3-Spam-Mail-Classification-by-NLP-and-ML

## Spam Detection Project

This project implements a Spam Detection system using a Naive Bayes classifier and text vectorization. Below is a step-by-step explanation of the project.

---

## Step 1: Importing Libraries
We import the necessary libraries for data handling and model building:
```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
```

---

## Step 2: Loading the Dataset
The dataset is loaded from a CSV file using `pandas.read_csv()`:
```python
data = pd.read_csv("spam.csv", encoding="latin-1")
```

### Initial Dataset Structure:
- **Columns**: `Category` (ham/spam), `Message` (text content), and additional unnamed columns.

---

## Step 3: Data Cleaning
Unnecessary columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`) are dropped:
```python
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True, errors='ignore')
```

The `Category` column is mapped to numeric values:
- `ham` → `0`
- `spam` → `1`
```python
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})
```

---

## Step 4: Checking the Dataset
Basic checks to ensure the data is clean and ready for processing:
1. Verify column names:
   ```python
   data.columns
   ```
2. Check for null values:
   ```python
   data.isnull().sum()
   ```

---

## Step 5: Splitting Data
The dataset is split into features (`X`) and labels (`Y`):
```python
X = data['Message']
Y = data['Category']
```

Shapes of `X` and `Y`:
```python
X.shape  # (5572,)
Y.shape  # (5572,)
```

---

## Step 6: Text Vectorization
Text data is converted into numerical form using `CountVectorizer`:
```python
cv = CountVectorizer()
X = cv.fit_transform(X)
```
This converts messages into a sparse matrix representation.

---

## Step 7: Training and Testing Split
The dataset is split into training and testing sets:
```python
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

Shape of the datasets:
- `x_train`: (4457, 8745)
- `x_test`: (1115, 8745)

---

## Step 8: Training the Model
A Naive Bayes classifier is initialized and trained on the training set:
```python
model = MultinomialNB()
model.fit(x_train, y_train)
```

---

## Step 9: Evaluating the Model
The model's accuracy is calculated on both the training and testing datasets:
```python
model.score(x_train, y_train)  # Training Accuracy: ~99.39%
model.score(x_test, y_test)    # Testing Accuracy: ~98.56%
```

---

## Step 10: Making Predictions
We test the model with a sample message:
```python
msg = "click the below link to get 5cor on your account"
data = [msg]
vect = cv.transform(data).toarray()
my_prediction = model.predict(vect)
```

The output (`array([1])`) indicates the message is classified as spam.

---

## Step 11: Saving the Model and Vectorizer
To reuse the model and vectorizer, they are saved using `pickle`:
```python
pickle.dump(model, open('spam123.pkl', 'wb'))
pickle.dump(cv, open('vec.pkl', 'wb'))
```

---

## Summary
This project demonstrates:
1. Data cleaning and preprocessing.
2. Converting text data into numerical features using `CountVectorizer`.
3. Training a Naive Bayes classifier for spam detection.
4. Saving the model for future use.

You can now use the saved model (`spam123.pkl`) and vectorizer (`vec.pkl`) to predict whether a message is spam or not.
