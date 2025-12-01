import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv(r"C:\Users\chamo\OneDrive\Desktop\coding\Python\Stock.csv")
data['Trend'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data['Trend']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
# Logistic Regression
log = LogisticRegression(
    max_iter=1000,random_state=42
)
# Decision Tree
tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=None
)
# Random Forest
forest = RandomForestClassifier(
    n_estimators=300,       # more trees
    max_depth=5,            # limit depth to reduce overfitting
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1               # use all cores
)
# 6. Train models
log.fit(X_train, y_train)
tree.fit(X_train, y_train)
forest.fit(X_train, y_train)
# 7. Predictions
pred_log = log.predict(X_test)
pred_tree = tree.predict(X_test)
pred_forest = forest.predict(X_test)
# 8. Accuracy scores
print("Logistic Regression Accuracy:", accuracy_score(y_test, pred_log))
print("Decision Tree Accuracy:", accuracy_score(y_test, pred_tree))
print("Random Forest Accuracy:", accuracy_score(y_test, pred_forest))
