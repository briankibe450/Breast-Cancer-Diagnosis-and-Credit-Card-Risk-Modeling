# Breast-Cancer-Diagnosis-and-Credit-Card-Risk-Modeling
This repository contains two comprehensive machine learning projects: binary classification of breast cancer patients using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, and credit card risk modeling using the German Credit dataset.

Project 1: Binary Classification of Breast Cancer Patients
Dataset
The dataset used for this project is the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, which can be found here.

Objective
Develop a binary classification model to predict whether a breast cancer case is malignant or benign.

Steps
Load and preprocess the dataset
Split the dataset into training and testing sets
Train a Logistic Regression model
Evaluate the model using accuracy, recall, precision, and ROC-AUC scores
Visualize the results
Code
python
Copy code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df = pd.read_csv(url, header=None, names=column_names)

# Drop the ID column
df.drop(columns=['ID'], inplace=True)

# Map Diagnosis to binary values
df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})

# Split the data into features and target
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Model Score:", model.score(X_test, y_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
Visualization
python
Copy code
# Confusion Matrix Heatmap
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
Project 2: Credit Card Risk Modeling
Dataset
The dataset used for this project is the German Credit dataset. The data contains observations on 30 variables for 1000 past applicants for credit.

Objective
Develop a credit-scoring rule to determine if a new applicant is a good credit risk or a bad credit risk.

Steps
Load and preprocess the dataset
Generate summary statistics and visualizations
Split the dataset into training, validation, and test sets
Train a Logistic Regression model
Evaluate the model using accuracy, recall, precision, and ROC-AUC scores
Visualize the results
Code
python
Copy code
# Load the dataset
df_credit = pd.read_csv('German.csv')

# Drop the first column
df_credit.drop(columns=[df_credit.columns[0]], inplace=True)

# Rename the columns
column_names = ["chk_acct", "duration", "credit_his", "purpose", "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", "present_resid", "property", "age", "other_install", "housing", "n_credits", "job", "n_people", "telephone", "foreign", "response"]
df_credit.columns = column_names

# Modify the 'response' variable
df_credit['response'] = df_credit['response'] - 1
df_credit['response'] = df_credit['response'].astype('object')

# Summarize the dataset
print(df_credit.info())
print(df_credit.describe(include='all'))

# Correlation matrix
corr_matrix = df_credit.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Bar plot for installment rate vs count of observations by response type
sns.countplot(x='installment_rate', hue='response', data=df_credit)
plt.title('Installment Rate vs Count by Response Type')
plt.show()

# Box plot for age by response variable
sns.boxplot(x='response', y='age', data=df_credit)
plt.title('Age by Response Variable')
plt.show()

# Box plot for duration by response variable
sns.boxplot(x='response', y='duration', data=df_credit)
plt.title('Duration by Response Variable')
plt.show()

# Bar plot for chk_acct vs count of observations by response type
sns.countplot(x='chk_acct', hue='response', data=df_credit)
plt.title('Checking Account vs Count by Response Type')
plt.show()

# Bar plot for credit_his vs count of observations by response type
sns.countplot(x='credit_his', hue='response', data=df_credit)
plt.title('Credit History vs Count by Response Type')
plt.show()

# Bar plot for saving_acct vs count of observations by response type
sns.countplot(x='saving_acct', hue='response', data=df_credit)
plt.title('Saving Account vs Count by Response Type')
plt.show()

# Split the data into X and y
X = df_credit[['sex', 'housing', 'saving_acct', 'chk_acct', 'age', 'duration', 'amount']]
y = df_credit['response']

# One hot encoding for categorical variables
X = pd.get_dummies(X, columns=['sex', 'housing', 'saving_acct', 'chk_acct'], drop_first=True)

# Standardize numerical columns
scaler = StandardScaler()
X[['age', 'duration', 'amount']] = scaler.fit_transform(X[['age', 'duration', 'amount']])

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
Part 3: SQL Table Creation and Queries
Objective
Create and manipulate SQL tables to store and query book and order data.

Steps
Create tables using SQLAlchemy
Insert data into tables
Perform SQL queries
Code
python
Copy code
from sqlalchemy import create_engine, Column, Integer, String, Float, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class BookCatalogue(Base):
    __tablename__ = 'book_catalogue'
    bookid = Column(Integer, primary_key=True)
    publisherid = Column(Integer)
    title = Column(String)
    author = Column(String)

class OrderTable(Base):
    __tablename__ = 'order_table'
    bookid = Column(Integer, primary_key=True)
    type = Column(String)
    author = Column(String)
    price = Column(Float)
    quantity = Column(Integer)

# Create an SQLite database and tables
engine = create_engine('sqlite:///books_orders.db')
Base.metadata.create_all(engine)

# Insert data into tables
Session = sessionmaker(bind=engine)
session = Session()

books_data = [
    BookCatalogue(bookid=348, publisherid=1, title='Autobiography', author='Malcolm X'),
    # (add other entries)
]

orders_data = [
    OrderTable(bookid=348, type='Autobiography', author='Malcolm X', price=523, quantity=8),
    # (add other entries)
]

session.add_all(books_data)
session.add_all(orders_data)
session.commit()
Conclusion
This repository contains two machine learning projects along with SQL table creation and querying. The projects cover data preprocessing, model training, evaluation, and visualization using Python libraries like pandas, scikit-learn, seaborn, and SQL
