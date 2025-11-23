# %%
import pandas as pd, numpy as np
from pathlib import Path

# %%
data_path = Path("./loan_approval_dataset.csv")
print("Dataset exists:", data_path.exists())

# %%
df = pd.read_csv(data_path)

# %%
df.shape, df.columns.tolist()


# %%
df.columns = df.columns.str.strip()

# %%
df.columns.tolist()

# %%
df.head()

# %%
df.info()

# %%
df.describe(include='all')

# %% [markdown]
# # EDA

# %%
import matplotlib.pyplot as plt, seaborn as sns
%matplotlib inline

# %%
df.columns.to_list()

# %%
target_col = 'loan_status'

# %%
print('Using target column:', target_col)
display(df[target_col].value_counts(dropna=False))
sns.countplot(x=target_col, data=df)
plt.title('Target distribution')
plt.show()

# %%
missing = df.isnull().sum().sort_values(ascending=False)
missing[missing>0]

# %%
target_col = "loan_status"
id_cols = ["loan_id"] 

# %%
df = df.drop(columns=id_cols, errors="ignore")

# %%
df_clf = df.dropna(subset=[target_col]).copy()

# %%
X = df_clf.drop(columns=[target_col])
y = df_clf[target_col].copy()

# %% [markdown]
# # Feature types & drop id

# %%
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# %%
print("Categorical columns:", cat_cols)
print("Numerical columns:", num_cols)

# %% [markdown]
# # Preprocessing pipeline

# %%
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# %%
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# %%
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# %%
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
], remainder='drop')

# %% [markdown]
# # Stage-1 : Classification baseline

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# %%
X.head()

# %%
y.head()

# %%
unique_vals = sorted(y.unique())

unique_vals

# %%
y = df_clf[target_col].astype(str).str.strip().str.lower()

# %%
unique_vals = sorted(y.unique())

unique_vals

# %%
y = (y == 'approved').astype(int)

# %%
y.head()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train/test sizes:", X_train.shape, X_test.shape)

# %% [markdown]
# # Fit RF baseline pipelines

# %%
# Random Forest baseline
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('rf', RandomForestClassifier(n_estimators=200, random_state=42, oob_score=True))])

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'rf__n_estimators': [100, 200, 300, 400],
    'rf__max_depth': [None, 4, 8, 10],
    'rf__max_features': ['sqrt', None]
}
grid = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params (RF):", grid.best_params_)
best_clf = grid.best_estimator_
y_pred_best = best_clf.predict(X_test)
print(classification_report(y_test, y_pred_best))

# %%
# Random Forest baseline
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('rf', RandomForestClassifier(n_estimators=200, max_depth=None, max_features='sqrt', random_state=42, oob_score=True))])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
print('Random Forest:')
print(classification_report(y_test, y_pred_rf))
print('RF OOB score (if available):', getattr(rf_pipeline.named_steps['rf'],'oob_score_', None))

# %%
rf_pipeline

# %% [markdown]
# # Stage 2 

# %%
# Numerical columns: ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']

# %%
df["loan_status"].unique()

# %%
df['loan_status'] = df['loan_status'].str.strip()

# %%
df['loan_status'].unique()

# %%
# Filter data for approved loans only
approved_df = df[df["loan_status"] == "Approved"].copy()

#Only return data with approved status

# %%
approved_df.head()

# %%
reg_target = "loan_amount"
reg_df = approved_df.dropna(subset=[reg_target]).copy()

# %%
X_reg = reg_df.drop(columns=[reg_target])
y_reg = reg_df[reg_target]

# %%
num_cols_reg = X_reg.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols_reg = X_reg.select_dtypes(include=['object']).columns.tolist()

# %%
print("Regression Numerical:", num_cols_reg)
print("Regression Categorical:", cat_cols_reg)

# %%
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# %%
reg_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols_reg),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols_reg)
    ]
)

# %%
from sklearn.ensemble import RandomForestRegressor

# %%
reg_pipe = Pipeline(steps=[
    ('preprocessor', reg_preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# %%
param_grid_reg = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5, 10]
}

# %%
grid_reg = GridSearchCV(
    reg_pipe,
    param_grid_reg,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

# Xr_train, Xr_test, yr_train, yr_test
grid_reg.fit(Xr_train, yr_train)

# %%
print("Best Regression Params:", grid_reg.best_params_)
best_reg = grid_reg.best_estimator_

# %%
grid_reg.best_score_

# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %%
reg_pipe = Pipeline(steps=[
    ('preprocessor', reg_preprocessor),
    ('model', RandomForestRegressor(max_depth=5, min_samples_split=10,n_estimators=300, random_state=42))
])

reg_pipe.fit(Xr_train, yr_train)

# Xr_train, Xr_test, yr_train, yr_test
yr_pred = reg_pipe.predict(Xr_test)
print("Regression metrics on approved applicants:")
print("RMSE:", mean_squared_error(yr_test, yr_pred))
print("MAE:", mean_absolute_error(yr_test, yr_pred))
print("R2:", r2_score(yr_test, yr_pred))

# %% [markdown]
# # Save artifacts and combined predict function

# %%
import joblib

# %%
joblib.dump(rf_pipeline, 'stage_1_rf_classifier_pipeline.pkl')
joblib.dump(reg_pipe, 'stage_2_rf_regression_pipeline.pkl')

# %% [markdown]
# # Prediction Function

# %%
clf = joblib.load('./stage_1_rf_classifier_pipeline.pkl')
reg = joblib.load('./stage_2_rf_regression_pipeline.pkl')

# %%
def two_stage_predict(applicant_df):
    out = {}

    # Stage 1
    approve = clf.predict(applicant_df)[0]
    print(approve)
    out['loan_status'] = int(approve)

    if approve == 1:
        # ADD predicted loan_status column to input
        applicant_df_reg = applicant_df.copy()
        applicant_df_reg['loan_status'] = 'Approve'

        # Stage 2 prediction
        pred = reg.predict(applicant_df_reg)[0]
        out['regression_prediction'] = float(pred)
    else:
        print("Not approved")

    return out


# %%
example_row = X_test.iloc[[0]]
print("Example row:")
display(example_row)


# %%
example_row.info()

# %%
print(reg.feature_names_in_)

# %%
print("Two-stage prediction:", two_stage_predict(example_row))

# %%
user_input = {
    "no_of_dependents": int(input("No. of Dependents: ")),
    "education": input("Education (Graduate/Not Graduate): "),
    "self_employed": input("Self Employed (Yes/No): "),
    "income_annum": float(input("Annual Income: ")),
    "loan_amount": float(input("Loan Amount Requested: ")),
    "loan_term": int(input("Loan Term (in years): ")),
    "cibil_score": int(input("CIBIL Score: ")),
    "residential_assets_value": float(input("Residential Assets Value: ")),
    "commercial_assets_value": float(input("Commercial Assets Value: ")),
    "luxury_assets_value": float(input("Luxury Assets Value: ")),
    "bank_asset_value": float(input("Bank Asset Value: "))
}

applicant_df = pd.DataFrame([user_input])

result = two_stage_predict(applicant_df)

print("Final Output:")
print(result)

# %%
list(user_input.keys())

# %%
try:
    print("FEATURE IMPORTANCES:")
    for col, score in zip(list(user_input.keys()), rf_pipeline.named_steps["rf"].feature_importances_):
        print(f"{col:30s} : {score:.4f}")
except:
    print("Cannot extract feature importance (maybe pipeline wrapped).")

# %%
sample = pd.DataFrame([{
    "no_of_dependents": 1,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 1200000,
    "loan_amount": 300000,
    "loan_term": 12,
    "cibil_score": 820,
    "residential_assets_value": 2000000,
    "commercial_assets_value": 500000,
    "luxury_assets_value": 0,
    "bank_asset_value": 550000
}])

print("Notebook pipeline:", rf_pipeline.predict(sample))
loaded = joblib.load('stage_1_rf_classifier_pipeline.pkl')
print("Saved pipeline:", loaded.predict(sample))
print("Saved predict_proba:", loaded.predict_proba(sample))

# %%



