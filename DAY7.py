#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------
# SAMPLE DATA (similar to Kaggle student dataset)
# -------------------------------------------------
df = pd.DataFrame({
    "hours_studied": [2, 5, 1, 3, 4, 6, 7, 8, 2, 9, 10, 11, 4, 5, 6],
    "past_grade":    [60, 65, 50, 55, 70, 75, 80, 82, 58, 85, 88, 90, 72, 73, 77],
    "absences":      [5, 3, 7, 4, 2, 1, 2, 0, 6, 1, 0, 0, 3, 2, 1],
    "marks":         [55, 68, 52, 60, 72, 80, 85, 88, 60, 92, 95, 97, 76, 78, 83]
})

# -------------------------------------------------
# STEP 1 — CLEAN DATA
# -------------------------------------------------
df.drop_duplicates(inplace=True)                   # remove duplicate rows
df.fillna(df.mean(numeric_only=True), inplace=True)  # fill missing numerics

# -------------------------------------------------
# STEP 2 — VISUALIZATIONS
# -------------------------------------------------
plt.figure(figsize=(5,4))
sns.histplot(df["hours_studied"], kde=True)
plt.title("Distribution of Hours Studied")
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=df["hours_studied"], y=df["marks"])
plt.title("Hours Studied vs Marks")
plt.show()

plt.figure(figsize=(5,4))
sns.boxplot(x=df["absences"], y=df["marks"])
plt.title("Absences vs Marks")
plt.show()

# -------------------------------------------------
# STEP 3 — BUILD REGRESSION MODEL
# -------------------------------------------------
X = df[["hours_studied", "past_grade", "absences"]]   # features
y = df["marks"]                                       # target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# -------------------------------------------------
# STEP 4 — INTERPRET COEFFICIENTS
# -------------------------------------------------
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print(coefficients)

# -------------------------------------------------
# STEP 5 — MODEL EVALUATION
# -------------------------------------------------
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R² Score :", r2_score(y_test, y_pred))


# In[5]:


import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# STEP 1 — DATASET → NUMPY ARRAYS
# -------------------------------------------------

# Data: Hours, Assignments_Submitted, Attendance, Marks
data = np.array([
    [1,  5, 70, 40],
    [2,  7, 75, 50],
    [3, 10, 80, 60],
    [4, 12, 85, 70],
    [5, 15, 90, 80]
], dtype=float)

# Features X (Hours, Assignments, Attendance) and Target y (Marks)
X = data[:, :3]   # shape: (5, 3)
y = data[:, 3]    # shape: (5,)

n_samples, n_features = X.shape

print("X (features):\n", X)
print("\ny (marks):\n", y)

# -------------------------------------------------
# STEP 2 — NORMALIZE FEATURES (Standardization)
# -------------------------------------------------

X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)

X_norm = (X - X_mean) / X_std

print("\nFeature means:", X_mean)
print("Feature stds :", X_std)
print("\nNormalized X:\n", X_norm)

# -------------------------------------------------
# STEP 3 — INITIALIZE PARAMETERS
# -------------------------------------------------

m = np.zeros(n_features)   # coefficients [m1, m2, m3]
c = 0.0                    # intercept

alpha = 0.1       # learning rate
epochs = 1000     # number of iterations

loss_history = []

# -------------------------------------------------
# STEP 4 — GRADIENT DESCENT
# -------------------------------------------------

for epoch in range(epochs):
    # Forward pass: prediction
    y_pred = X_norm.dot(m) + c   # shape: (5,)

    # Error
    error = y_pred - y           # shape: (5,)

    # Mean Squared Error (loss)
    loss = (error ** 2).mean()
    loss_history.append(loss)

    # Gradients (vectorized)
    grad_m = (2 / n_samples) * X_norm.T.dot(error)  # shape: (3,)
    grad_c = (2 / n_samples) * error.sum()

    # Parameter update
    m = m - alpha * grad_m
    c = c - alpha * grad_c

    # Optional: print some epochs
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss = {loss:.4f}")

print("\nFinal coefficients (m):", m)
print("Final intercept (c):", c)

# -------------------------------------------------
# STEP 5 — PLOT LOSS VS EPOCHS
# -------------------------------------------------

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss vs Epochs (Gradient Descent)")
plt.grid(True)
plt.show()

# -------------------------------------------------
# STEP 6 — PREDICTION FUNCTION
# -------------------------------------------------

def predict(hours, assignments, attendance):
    """
    Predict marks for a student given:
    - hours: study hours
    - assignments: number of assignments submitted
    - attendance: attendance percentage
    """
    x = np.array([hours, assignments, attendance], dtype=float)
    # Normalize with training mean/std
    x_norm = (x - X_mean) / X_std
    y_hat = x_norm.dot(m) + c
    return y_hat

# Predict for:
# 1) 4 hours, 12 assignments, 85% attendance
# 2) 6 hours, 15 assignments, 92% attendance

student1 = predict(4, 12, 85)
student2 = predict(6, 15, 92)

print(f"\nPredicted marks for (4h, 12 assignments, 85% attendance): {student1:.2f}")
print(f"Predicted marks for (6h, 15 assignments, 92% attendance): {student2:.2f}")


# In[7]:


# ================================
# IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================
# LOAD DATA INTO DATAFRAME
# ================================
df = pd.DataFrame({
    "Income": [45000, 65000, 32000, 55000, 72000],
    "Family_Size": [4, 5, 3, 4, 6],
    "Online_Orders": [12, 8, 5, 10, 15],
    "Monthly_Grocery_Spend": [5800, 7200, 4100, 6500, 8800]
})

print("📌 Initial Data:")
print(df)

# ================================
# STEP 1 — EDA
# ================================
print("\n📊 Summary Statistics:\n", df.describe())

# Distribution plots
fig, axs = plt.subplots(1, 3, figsize=(15,4))
sns.histplot(df["Income"], kde=True, ax=axs[0])
axs[0].set_title("Income Distribution")

sns.histplot(df["Online_Orders"], kde=True, ax=axs[1])
axs[1].set_title("Online Orders Distribution")

sns.histplot(df["Monthly_Grocery_Spend"], kde=True, ax=axs[2])
axs[2].set_title("Grocery Spend Distribution")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Scatter plots
plt.figure(figsize=(5,4))
sns.scatterplot(x=df["Income"], y=df["Monthly_Grocery_Spend"])
plt.title("Income vs Grocery Spending")
plt.show()

plt.figure(figsize=(5,4))
sns.scatterplot(x=df["Online_Orders"], y=df["Monthly_Grocery_Spend"])
plt.title("Online Orders vs Grocery Spending")
plt.show()

# ================================
# STEP 2 — DATA CLEANING
# ================================
# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values (none here, but good practice)
df.fillna(df.mean(numeric_only=True), inplace=True)

# ================================
# STEP 3 — MODEL BUILDING
# ================================
X = df[["Income", "Family_Size", "Online_Orders"]]
y = df["Monthly_Grocery_Spend"]

# Train-test split (2 samples in test set → R² is valid)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train
X_test_scaled = scaler.transform(X_test)         # transform test

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions on test data
y_pred = model.predict(X_test_scaled)

# ================================
# STEP 4 — EVALUATION + FEATURE IMPORTANCE
# ================================
print("\n📊 Model Evaluation Metrics:")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R²  :", r2_score(y_test, y_pred))

# Coefficient for each feature
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\n📌 Feature Importance (Higher Coefficient = More Influence):")
print(coeff_df)

# =================


# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ===============================
# LOAD DATA
# ===============================
df = pd.DataFrame({
    "Temperature": [28, 30, 26, 32, 27],
    "Humidity": [60, 65, 55, 70, 58],
    "Appliances_On": [4, 5, 3, 6, 4],
    "Energy_Consumed": [220, 250, 180, 300, 210]
})

print(df)

# ===============================
# CORRELATION ANALYSIS
# ===============================
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

corr = df.corr()["Energy_Consumed"].sort_values(ascending=False)
print("\n🔍 Correlation with Energy Consumption:")
print(corr)

# Highest correlation feature
highest_feature = corr.index[1]  # 0th is Energy_Consumed itself
print("\n🔥 Highest correlated feature →", highest_feature)

# ===============================
# MODEL TRAINING
# ===============================
X = df[["Temperature", "Humidity", "Appliances_On"]]
y = df["Energy_Consumed"]

# Split (40% → 2 test samples so R² is valid)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Scale values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ===============================
# PREDICTION FOR NEW INPUT
# New case:
# Temperature = 31°C, Humidity = 63%, Appliances_On = 5
# ===============================
new_input = pd.DataFrame([[31, 63, 5]],
                         columns=["Temperature", "Humidity", "Appliances_On"])

new_scaled = scaler.transform(new_input)
prediction = model.predict(new_scaled)[0]

print("\n🔋 Predicted Energy Consumption:")
print(f"➡ For 31°C, 63% humidity, 5 appliances → {round(prediction, 2)} units")


# In[9]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================================
# SAMPLE STARTUP DATA
# ======================================
df = pd.DataFrame({
    "Marketing_Spend": [10000, 15000, 12000, 25000, 30000, 80000],   # Outlier = 80k
    "Employees":       [10, 15, 12, 25, 30, 50],                    # Outlier = 50
    "Product_Price":   [200, 220, 210, 230, 240, 300],              # Outlier = 300
    "Revenue":         [40000, 52000, 46000, 78000, 90000, 120000]  # Outlier = 120k
})

print("📌 Initial Data:")
print(df)

# ======================================
# STEP 1 — EDA
# ======================================
print("\n📊 Summary Statistics:\n", df.describe())

# Heatmap correlation
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df)
plt.show()

# ======================================
# STEP 2 — OUTLIER DETECTION
# (Using IQR method)
# ======================================
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Mask outliers
mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)

df_clean = df[mask]
print("\n🧹 After removing outliers:")
print(df_clean)

# ======================================
# STEP 3 — MODEL FITTING
# ======================================
X = df_clean[["Marketing_Spend", "Employees", "Product_Price"]]
y = df_clean["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# ======================================
# STEP 4 — EVALUATION
# ======================================
print("\n📈 Model Performance:")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R²  :", r2_score(y_test, y_pred))

# ======================================
# STEP 5 — COEFFICIENTS
# ======================================
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\n📌 Feature Influence:")
print(coef_df)


# In[10]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================
# DATASET
# ==========================
df = pd.DataFrame({
    "RAM": [8, 16, 8, 32, 16],
    "Storage": [512, 1024, 256, 1024, 512],
    "Processor_Speed": [2.8, 3.2, 2.5, 3.6, 3.0],
    "Brand": ["Dell", "HP", "Lenovo", "Apple", "Asus"],
    "Price": [58000, 82000, 52000, 150000, 78000]
})

print("📌 Original Data:")
print(df)

# ==========================
# STEP 1 — ENCODE BRAND
# ==========================
df_encoded = pd.get_dummies(df, columns=["Brand"], drop_first=True)
print("\n🔠 Encoded DataFrame:")
print(df_encoded)

# Separate input and output
X = df_encoded.drop("Price", axis=1)
y = df_encoded["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# ==========================
# STEP 2 — MODEL WITHOUT SCALING
# ==========================
model_raw = LinearRegression()
model_raw.fit(X_train, y_train)
y_pred_raw = model_raw.predict(X_test)

mse_raw = mean_squared_error(y_test, y_pred_raw)
r2_raw = r2_score(y_test, y_pred_raw)

# ==========================
# STEP 3 — MODEL WITH SCALING
# ==========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)

mse_scaled = mean_squared_error(y_test, y_pred_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

# ==========================
# STEP 4 — COMPARE PERFORMANCE
# ==========================
print("\n📊 MODEL PERFORMANCE COMPARISON")
print("----------------------------------------")
print(f"MSE with NO scaling : {mse_raw:.2f}")
print(f"R²  with NO scaling : {r2_raw:.4f}")
print("----------------------------------------")
print(f"MSE WITH scaling    : {mse_scaled:.2f}")
print(f"R²  WITH scaling    : {r2_scaled:.4f}")
print("----------------------------------------")

# ==========================
# SHOW COEFFICIENTS (OPTIONAL)
# ==========================
print("\n📌 Scaled Model Coefficients:")
for feature, coef in zip(X.columns, model_scaled.coef_):
    print(f"{feature:20s} → {coef:.4f}")


# In[11]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================
# LOAD IRIS DATA
# ============================================
iris = sns.load_dataset("iris")
print("📌 First 5 rows of dataset:")
print(iris.head())

# ============================================
# STEP 1 — SELECT TARGET & FEATURES
# Target  = sepal_length
# Features = remaining numeric columns
# ============================================
X = iris[["sepal_width", "petal_length", "petal_width"]]
y = iris["sepal_length"]

# ============================================
# TRAIN–TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# FIT REGRESSION MODEL
# ============================================
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ============================================
# STEP 2 — EVALUATE MODEL PERFORMANCE
# ============================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"R²  : {r2:.4f}")

# ============================================
# STEP 3 — PLOT RESIDUALS
# Residual = actual − predicted
# ============================================
residuals = y_test - y_pred

plt.figure(figsize=(7,5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Sepal Length")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot — Iris Regression Model")
plt.show()


# In[12]:


# ============================================
# PART 4 — ALGORITHMIC THINKING EXERCISES
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make plots look nicer
plt.style.use("default")

# --------------------------------------------
# 1. NUMPY-BASED ALGORITHM: FEATURE NORMALIZATION
# --------------------------------------------

def normalize_feature(x):
    """
    Normalize a 1D numpy array or pandas Series:
    (x - mean) / std
    """
    x = np.array(x, dtype=float)
    return (x - x.mean()) / x.std()

# Example usage for hours, assignments, attendance
hours = np.array([1, 2, 3, 4, 5])
assignments = np.array([5, 7, 10, 12, 15])
attendance = np.array([70, 75, 80, 85, 90])

hours_norm = normalize_feature(hours)
assignments_norm = normalize_feature(assignments)
attendance_norm = normalize_feature(attendance)

print("=== 1) NORMALIZATION EXAMPLE ===")
print("Original hours:     ", hours)
print("Normalized hours:   ", np.round(hours_norm, 3))
print("Original assignments:", assignments)
print("Normalized assignments:", np.round(assignments_norm, 3))
print("Original attendance:", attendance)
print("Normalized attendance:", np.round(attendance_norm, 3))
print("\n")


# --------------------------------------------
# 2. PANDAS ALGORITHM: REMOVE OUTLIERS (IQR)
# --------------------------------------------

def remove_outliers_iqr(df, columns=None):
    """
    Remove rows that have outliers in the given columns using IQR method.
    Outliers: values < Q1 - 1.5*IQR or > Q3 + 1.5*IQR
    If columns=None, apply on all numeric columns.
    Returns: cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Keep only rows within bounds
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    return df_clean

# Example dataset with outliers
df_outlier_example = pd.DataFrame({
    "Marks": [60, 65, 70, 72, 68, 95, 1000],   # 1000 is an outlier
    "Hours": [2, 3, 4, 5, 3, 6, 100]          # 100 is an outlier
})

print("=== 2) OUTLIER REMOVAL (IQR) ===")
print("Original DataFrame:")
print(df_outlier_example)

df_no_outliers = remove_outliers_iqr(df_outlier_example, columns=["Marks", "Hours"])
print("\nAfter removing outliers:")
print(df_no_outliers)
print("\n")


# --------------------------------------------
# 3. MINI EDA ALGORITHM
# --------------------------------------------

def mini_eda(df, target_col=None):
    """
    Given any DataFrame:
      • Detect missing values
      • Detect duplicates
      • Compute correlations (numeric)
      • Identify top 3 strongest features (by correlation with target_col)

    If target_col is None, strongest features are chosen based on
    average absolute correlation with all other numeric features.
    """
    print("=== MINI EDA REPORT ===\n")

    # 1) Missing values
    print("1) Missing values per column:")
    print(df.isna().sum())
    print("\n")

    # 2) Duplicates
    num_duplicates = df.duplicated().sum()
    print(f"2) Number of duplicate rows: {num_duplicates}")
    print("\n")

    # 3) Correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        print("No numeric columns to compute correlation.")
        return

    corr_matrix = numeric_df.corr()
    print("3) Correlation matrix (numeric columns):")
    print(corr_matrix)
    print("\n")

    # 4) Top 3 strongest features
    print("4) Top 3 strongest features:")
    if target_col is not None and target_col in numeric_df.columns:
        # Correlation with target
        target_corr = corr_matrix[target_col].drop(labels=[target_col])
        strongest = target_corr.abs().sort_values(ascending=False).head(3)
        print(f"(Based on correlation with target: '{target_col}')")
        print(strongest)
    else:
        # Average absolute correlation with others
        abs_corr = corr_matrix.abs()
        mean_corr = abs_corr.mean().sort_values(ascending=False)
        strongest = mean_corr.head(3)
        print("(Based on average absolute correlation with other numeric features)")
        print(strongest)
    print("\n")

# Example dataset for mini_eda
df_student = pd.DataFrame({
    "hours": [2, 3, 4, 5, 3],
    "assignments": [5, 7, 10, 12, 7],
    "attendance": [70, 75, 80, 85, 75],
    "marks": [55, 60, 70, 80, 62]
})

mini_eda(df_student, target_col="marks")


# --------------------------------------------
# 4. VISUALIZATION ALGORITHM
# --------------------------------------------

def plot_basic_eda(df):
    """
    Given any DataFrame:
      • Plots histogram for numeric features
      • Plots boxplot for numeric features
      • Plots countplot for categorical features
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    # Histograms for numeric columns
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.show()

    # Boxplots for numeric columns
    for col in numeric_cols:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)
        plt.show()

    # Countplots for categorical columns
    for col in cat_cols:
        plt.figure()
        sns.countplot(x=df[col])
        plt.title(f"Countplot of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

# Example dataset with a categorical column
df_vis = pd.DataFrame({
    "Brand": ["Dell", "HP", "Dell", "Lenovo", "HP", "Apple"],
    "Price": [50000, 60000, 52000, 58000, 63000, 120000],
    "RAM": [8, 8, 16, 8, 16, 16]
})

print("=== 4) RUNNING VISUALIZATION ALGORITHM (CHECK PLOTS) ===")
plot_basic_eda(df_vis)


# In[13]:


# ============================================
# UBER FARE PREDICTION – END-TO-END EXAMPLE
# ============================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. CREATE SAMPLE DATA
# Each row = one Uber ride
# -----------------------------
df = pd.DataFrame({
    "distance_km":      [3.2, 5.5, 10.0, 2.0, 8.3, 4.1, 12.5, 6.0, 9.7, 1.8],
    "traffic_level":    ["low", "medium", "high", "low", "high", "medium", "high", "medium", "high", "low"],
    "hour_of_day":      [9, 18, 21, 11, 8, 17, 22, 19, 7, 14],   # 0-23
    "surge_multiplier": [1.0, 1.2, 1.5, 1.0, 1.3, 1.1, 1.8, 1.3, 1.6, 1.0],
    "car_type":         ["UberGo", "UberX", "UberX", "UberGo", "UberXL", "UberGo", "UberXL", "UberX", "UberXL", "UberGo"],
    "fare_amount":      [120, 180, 320, 90, 260, 150, 400, 210, 330, 80]
})

print("=== Sample Uber Data ===")
print(df.head())

# -----------------------------
# 2. BASIC EDA
# -----------------------------
print("\n=== Info ===")
print(df.info())

print("\n=== Summary Statistics (Numeric) ===")
print(df.describe())

# Correlation only on numeric columns
plt.figure(figsize=(6,4))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

# Optional: scatter distance vs fare
plt.figure(figsize=(5,4))
sns.scatterplot(x=df["distance_km"], y=df["fare_amount"])
plt.title("Distance vs Fare")
plt.xlabel("Distance (km)")
plt.ylabel("Fare amount")
plt.show()

# -----------------------------
# 3. PREPROCESSING
#    - Encode categorical columns
#    - Define X (inputs) and y (output)
# -----------------------------

# One-hot encode 'traffic_level' and 'car_type'
df_encoded = pd.get_dummies(df, columns=["traffic_level", "car_type"], drop_first=True)

print("\n=== Encoded Data ===")
print(df_encoded.head())

# Features and target
X = df_encoded.drop("fare_amount", axis=1)
y = df_encoded["fare_amount"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)

# -----------------------------
# 4. MODEL 1 – WITHOUT SCALING
# -----------------------------
model_raw = LinearRegression()
model_raw.fit(X_train, y_train)

y_pred_raw = model_raw.predict(X_test)

mae_raw = mean_absolute_error(y_test, y_pred_raw)
mse_raw = mean_squared_error(y_test, y_pred_raw)
rmse_raw = np.sqrt(mse_raw)
r2_raw = r2_score(y_test, y_pred_raw)

print("\n=== Model WITHOUT Scaling ===")
print("MAE  :", round(mae_raw, 2))
print("RMSE :", round(rmse_raw, 2))
print("R²   :", round(r2_raw, 4))

# -----------------------------
# 5. MODEL 2 – WITH SCALING
#    (Scale only numeric features)
# -----------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = model_scaled.predict(X_test_scaled)

mae_scaled = mean_absolute_error(y_test, y_pred_scaled)
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
rmse_scaled = np.sqrt(mse_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

print("\n=== Model WITH Scaling ===")
print("MAE  :", round(mae_scaled, 2))
print("RMSE :", round(rmse_scaled, 2))
print("R²   :", round(r2_scaled, 4))

# -----------------------------
# 6. COMPARE MODELS
# -----------------------------
print("\n=== COMPARISON (Raw vs Scaled) ===")
print(f"MAE  Raw    : {mae_raw:.2f}")
print(f"MAE  Scaled : {mae_scaled:.2f}")
print(f"R²   Raw    : {r2_raw:.4f}")
print(f"R²   Scaled : {r2_scaled:.4f}")

# -----------------------------
# 7. PREDICT FARE FOR NEW RIDE
# Example:
# distance = 7.0 km
# traffic = 'high'
# hour_of_day = 19
# surge_multiplier = 1.5
# car_type = 'UberX'
# -----------------------------

new_ride = pd.DataFrame({
    "distance_km": [7.0],
    "hour_of_day": [19],
    "surge_multiplier": [1.5],
    "traffic_level_high": [1],    # high
    "traffic_level_low": [0],     # not low
    "traffic_level_medium": [0],  # not medium (depends on your dummies)
    "car_type_UberGo": [0],
    "car_type_UberX": [1],
    "car_type_UberXL": [0]
}, columns=X.columns)  # ensure same column order as X

# Scale using same scaler
new_ride_scaled = scaler.transform(new_ride)

predicted_fare = model_scaled.predict(new_ride_scaled)[0]
print("\n=== Predicted Fare for New Ride ===")
print("Features:", new_ride.iloc[0].to_dict())
print("Predicted fare ≈", round(predicted_fare, 2))


# In[14]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# SAMPLE DATASET (10 Houses)
# ==========================================
df = pd.DataFrame({
    "num_ACs": [1, 2, 1, 3, 2, 3, 4, 1, 2, 3],
    "avg_daily_usage_hours": [5, 6, 4, 9, 7, 10, 12, 4, 6, 8],
    "meter_reading": [230, 350, 200, 450, 380, 500, 590, 210, 300, 470],
    "month": ["Jan", "Mar", "Jun", "May", "Apr", "Jul", "Aug", "Feb", "Jun", "May"],
    "num_people": [3, 4, 2, 5, 4, 6, 5, 2, 3, 4],
    "electric_bill": [1800, 2800, 1600, 4000, 3300, 4600, 5200, 1700, 2500, 4100]
})

print("📌 Initial Data:")
print(df)

# ==========================================
# ENCODE MONTH (Seasonal Effect)
# ==========================================
df_encoded = pd.get_dummies(df, columns=["month"], drop_first=True)

# Inputs & output
X = df_encoded.drop("electric_bill", axis=1)
y = df_encoded["electric_bill"]

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==========================================
# SCALE NUMERIC FEATURES
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# TRAIN REGRESSION MODEL
# ==========================================
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# ==========================================
# EVALUATION
# ==========================================
print("\n📊 Model Evaluation:")
print("MAE :", round(mean_absolute_error(y_test, y_pred), 2))
print("MSE :", round(mean_squared_error(y_test, y_pred), 2))
print("R²  :", round(r2_score(y_test, y_pred), 4))

# ==========================================
# FEATURE IMPORTANCE (Coefficients)
# ==========================================
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\n🔍 Feature Importance:")
print(coeff_df)


# In[15]:


# ============================================
# CASE STUDY 3 – PREDICT RESTAURANT SALES
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------
# 1. CREATE SAMPLE DATA
# --------------------------------------------
# Each row = one day’s data for a restaurant
df = pd.DataFrame({
    "day_of_week": [
        "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
        "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"
    ],
    "weather": [
        "Sunny", "Cloudy", "Rainy", "Sunny", "Rainy", "Sunny", "Cloudy",
        "Rainy", "Sunny", "Cloudy", "Sunny", "Rainy", "Sunny", "Cloudy"
    ],
    "offers": [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1],  # 0 = no offer, 1 = offer
    "customer_rating": [4.0, 3.8, 3.5, 4.2, 4.0, 4.5, 4.6, 3.6, 4.1, 3.7, 4.3, 3.4, 4.7, 4.4],
    "sales": [1200, 1500, 900, 1300, 1600, 2200, 2100,
              1100, 1550, 950, 1700, 1000, 2300, 2000]
})

print("=== Raw Data ===")
print(df.head())

# --------------------------------------------
# 2. BASIC EDA – WHAT AFFECTS SALES?
# --------------------------------------------
print("\n=== Summary Statistics ===")
print(df.describe())

# Average sales by day of week
print("\n=== Average Sales by Day of Week ===")
print(df.groupby("day_of_week")["sales"].mean())

# Average sales by weather
print("\n=== Average Sales by Weather ===")
print(df.groupby("weather")["sales"].mean())

# Pairplot to see relationships
sns.pairplot(df, hue="day_of_week")
plt.show()

# --------------------------------------------
# 3. ENCODING CATEGORICAL FEATURES
# --------------------------------------------
# One-hot encode 'day_of_week' and 'weather'
df_encoded = pd.get_dummies(df, columns=["day_of_week", "weather"], drop_first=True)

print("\n=== Encoded Data ===")
print(df_encoded.head())

# --------------------------------------------
# 4. DEFINE FEATURES (X) AND TARGET (y)
# --------------------------------------------
X = df_encoded.drop("sales", axis=1)
y = df_encoded["sales"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------------------------
# 5. SCALE FEATURES (OPTIONAL BUT GOOD)
# --------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------
# 6. TRAIN REGRESSION MODEL
# --------------------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# --------------------------------------------
# 7. EVALUATE MODEL
# --------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== Model Performance ===")
print("MAE  :", round(mae, 2))
print("RMSE :", round(rmse, 2))
print("R²   :", round(r2, 4))

# --------------------------------------------
# 8. FEATURE IMPORTANCE (COEFFICIENTS)
# --------------------------------------------
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\n=== Feature Importance (Higher = more positive effect on sales) ===")
print(coeff_df)

# --------------------------------------------
# 9. QUICK CHECK: WHICH FEATURES MAY BE LESS IMPORTANT?
#    (Very small coefficients ≈ less impact)
# --------------------------------------------
print("\nPossible less important features (small coefficients):")
print(coeff_df.tail())


# In[ ]:




