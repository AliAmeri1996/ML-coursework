from sklearn.datasets import make_classification #Creates a fake dataset for testing classification models
from sklearn.linear_model import LogisticRegression #The ML model (predicts binary classes 0 or 1)
from sklearn.model_selection import train_test_split #Splits data into training and test sets
from sklearn.pipeline import make_pipeline #Chains multiple preprocessing + model steps together
from sklearn.preprocessing import StandardScaler #Standardizes features (mean=0, std=1)

X, y = make_classification(random_state=42)
'''This creates an artificial dataset with:

X: feature matrix (by default 100 samples × 20 features)

y: binary labels (0 or 1)
Basically, fake data that looks like a real classification problem.'''

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
'''X_train, y_train: used to train the model (75%)

X_test, y_test: used to test the model (25%)

Setting random_state=42 ensures reproducibility.'''

pipe = make_pipeline(StandardScaler(), LogisticRegression())
'''| Step | Operation              | Description                               |
| ---- | ---------------------- | ----------------------------------------- |
| 1️⃣  | `StandardScaler()`     | Standardizes features (mean = 0, std = 1) |
| 2️⃣  | `LogisticRegression()` | Trains the logistic regression classifier |
'''

'''So whenever you call pipe.fit(), it will:

Fit the scaler on your training data,

Transform the training data,

Train the logistic regression on that scaled data.'''


pipe.fit(X_train, y_train)  # apply scaling on training data,fit() learns from the data
'''This performs both preprocessing and model training in one step:

The StandardScaler learns the mean & std of X_train

It scales X_train → X_train_scaled

The LogisticRegression then learns from X_train_scaled and y_train

✅ All this happens inside the pipeline, so it’s clean and safe — no manual scaling mistakes.'''

pipe.score(X_test, y_test)  # apply scaling on testing data, without leaking training data.
'''When you call .score() on the pipeline:

The StandardScaler transforms X_test using the same means/stds learned from the training data (no data leakage!)

The trained logistic regression predicts labels for that scaled X_test

.score() returns the accuracy (fraction of correct predictions)'''


print("Model accuracy on test set:", pipe.score(X_test, y_test))
