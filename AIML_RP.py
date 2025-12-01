import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = pd.read_csv(r"C:\Users\chamo\OneDrive\Desktop\coding\Python\Stock.csv")   # contains Open, High, Low, Close, Volume
# Create Trend label: UP if next day's close > today's close
data['Trend'] = (data['Close'].shift(-1) > data['Close']).astype(int)
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features][:-1]
y = data['Trend'][:-1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train, y_train)
pred1 = log.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, pred1))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
pred2 = tree.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, pred2))

#Random Forest (Best Method)
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
pred3 = forest.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, pred3))