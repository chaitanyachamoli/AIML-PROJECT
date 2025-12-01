import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv(r"C:\Users\chamo\OneDrive\Desktop\coding\Python\Stock.csv")
# Create Trend column (1 = price up next day, 0 = price down)
data['Trend'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data['Trend']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)
# Logistic Regression model
log = LogisticRegression(max_iter=1000, random_state=42)
# Random Forest model 
forest = RandomForestClassifier(
    n_estimators=300,       
    max_depth=5,            
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
# Train models
log.fit(X_train, y_train)
forest.fit(X_train, y_train)
# Predictions
pred_log = log.predict(X_test)
pred_forest = forest.predict(X_test)
# Accuracy scores
print("Logistic Regression Accuracy:", accuracy_score(y_test, pred_log))
print("Random Forest Accuracy:", accuracy_score(y_test, pred_forest))
