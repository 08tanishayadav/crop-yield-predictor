import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
import pickle

# Load dataset
pstore = pd.read_csv(r"C:\Users\Tanisha Yadav\Downloads\crop_yield-1.csv")

# Check for missing values
print(pstore.isnull().sum())

# Drop rows with missing values
pstore = pstore.dropna()

# Convert numeric columns safely
pstore.loc[:, 'Annual_Rainfall'] = pd.to_numeric(pstore['Annual_Rainfall'], errors='coerce')
pstore.loc[:, 'Fertilizer'] = pd.to_numeric(pstore['Fertilizer'], errors='coerce')
pstore.loc[:, 'Pesticide'] = pd.to_numeric(pstore['Pesticide'], errors='coerce')
pstore.loc[:, 'Yield'] = pd.to_numeric(pstore['Yield'], errors='coerce')
pstore.loc[:, 'Area'] = pd.to_numeric(pstore['Area'], errors='coerce')

# Filter necessary columns
pstore = pstore[['Crop', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']]

# One-hot encode categorical columns
df_encoded = pd.get_dummies(pstore[['Crop', 'Season', 'State']])

# Combine features and target
X_features = pstore[['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
x = pd.concat([df_encoded, X_features], axis=1)
y = pstore['Yield']

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Define hyperparameter grid for tuning
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = make_scorer(r2_score)

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(),
    param_distributions=param_dist,
    n_iter=10,
    cv=cv,
    scoring=scoring,
    random_state=42
)

random_search.fit(X_train, Y_train)

# Retrieve the best model
best_model = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)

# Evaluate the model
y_pred = best_model.predict(X_test)
print(f"Optimized MAE: {mean_absolute_error(Y_test, y_pred):.2f}")
print(f"Optimized R²: {r2_score(Y_test, y_pred):.2f}")

cv_scores = cross_val_score(best_model, x, y, cv=cv, scoring='r2')
print(f"Cross-Validated R²: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.2f}")

# Save model and columns
with open('crop_yield_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('model_columns.pkl', 'wb') as columns_file:
    pickle.dump(list(x.columns), columns_file)

# --- COMMENTED OUT EXTRA CODE BELOW ---

# print(pstore)
# category_data = pstore.groupby('Crop')[['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']].mean().round(3)
# print(category_data)
# coorelation = pstore[['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']].corr()
# print(coorelation)
# plt.figure(figsize=(5,4))
# sns.heatmap(coorelation, annot=True, cmap='coolwarm')
# plt.title("COORELATION HEATMAP")
# plt.show()
# crops = [...]
# seasons = [...]
# states = [...]
# product = list(itertools.product(crops, seasons, states))
# df = pd.DataFrame(product, columns=['Crop','Season','State'])
# print(df)
# df_encoded = pd.get_dummies(df, columns=['Crop','Season','State'])
# print(df_encoded)
# x = df_encoded.drop('Yield', axis=1)
# y = df_encoded['Yield']
# assert len(df_encoded)==len(pstore)
# df_encoded.columns
# df_encoded = df_encoded.reset_index(drop=True)
# pstore = pstore.reset_index(drop=True)
# assert len(df_encoded) == len(pstore)
# print(f"df_encoded shape: {df_encoded.shape}")
# print(f"pstore shape: {pstore.shape}")