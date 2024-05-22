import pandas as pd
import numpy as np
from lib import EDA as eda, Feng as feng
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

eda.set_display()

df = pd.read_csv('dataset/car_dataset.csv')

df['RPM'] = df['torque'].str.extract(r'(\d+)rpm', expand=False)
df['RPM'].fillna(df['torque'].str.extract(r'(\d{1,3}(?:,\d{3})*)\(kgm@ rpm\)', expand=False), inplace=True)
df['RPM'].fillna(df['torque'].str.extract(r'(\d+) RPM', expand=False), inplace=True)
df['RPM'].fillna(df['torque'].str.extract(r'(\d+)  rpm ', expand=False), inplace=True)
df['RPM'].fillna(df['torque'].str.extract(r'(\d+) rpm', expand=False), inplace=True)

df['TORQUE'] = df['torque'].str.extract(r'(\d+)Nm@', expand=False)
df['TORQUE'].fillna(df['torque'].str.extract(r'(\d+)nm@', expand=False), inplace=True)
df['TORQUE'].fillna(df['torque'].str.extract(r'(\d+) Nm', expand=False), inplace=True)
df['TORQUE'].fillna(df['torque'].str.extract(r'(\d+)@', expand=False), inplace=True)
df['TORQUE'].fillna(df['torque'].str.extract(r'(\d+)Nm', expand=False), inplace=True)
df['TORQUE'].fillna(df['torque'].str.extract(r'(\d+)  Nm', expand=False), inplace=True)
df['TORQUE'].fillna(df['torque'].str.extract(r'(\d+)NM@', expand=False), inplace=True)
df['TORQUE'].fillna(
    (df['torque'].str.extract(r'(\d{1,2}(?:[,.]\d{1,2})?)@\s*\d{1,3}(?:,\d{3})*\(kgm@ rpm\)', expand=False)).astype(
        float) * 9.8, inplace=True)
df['TORQUE'].fillna((df['torque'].str.extract(r'(\d{1,2}(?:[,.]\d{1,2})?)\s*kgm', expand=False)).astype(float) * 9.8,
                    inplace=True)

df['RPM'] = df['RPM'].str.replace(',', '').astype(float)
df['TORQUE'] = df['TORQUE'].astype(float)

df['BRAND'] = df['name'].astype('str').apply(lambda x: x.split()[0])

df['MILEAGE'] = (df['mileage'].apply(lambda x: str(x).replace(' kmpl', '').replace(' km/kg', ''))).astype(float)

df['ENGINE'] = (df['engine'].apply(lambda x: str(x).replace(' CC', ''))).astype(float)

df['MAX_POWER'] = df['max_power'].astype(str).apply(lambda x: x.split()[0])

df.drop(df[df['MAX_POWER'] == 'bhp'].index, inplace=True, axis=0)

df['MAX_POWER'] = df['MAX_POWER'].astype(float)

df.drop(['torque', 'name', 'mileage', 'engine', 'max_power'], axis=1, inplace=True)

result = eda.grab_col_names(df)
cat_cols, num_cols = result[0], result[1]

eda.check_df(df)

for col in num_cols:
    eda.num_summary(df, col)

for col in cat_cols:
    eda.cat_summary(df, col)

eda.plot_numerical_col(df, num_cols)

eda.plot_categoric_col(df, cat_cols)

eda.high_correlated_cols(df, num_cols, plot=True)

for col in cat_cols:
    eda.target_summary_with_cat(df, 'selling_price', col)

for col in num_cols:
    eda.target_summary_with_num(df, 'selling_price', col)

eda.rare_analyser(df, 'selling_price', cat_cols)

eda.missing_values_table(df)

eda.missing_values_heatmap_grap(df)

######################################################################


bins = [29999, 100000, 300000, 500000, 1000000, np.inf]
labels = ['<100K', '100K-300K', '300K-500K', '500K-1M', '1M>']

df['PRICE_CAT'] = pd.cut(df['selling_price'], bins=bins, labels=labels, include_lowest=True)

miss_list = ['RPM', 'TORQUE', 'MILEAGE', 'ENGINE', 'MAX_POWER']

for col in miss_list:
    df[col] = df[col].apply(lambda x: np.nan if x == 0 else x)
    df[col].fillna(df.groupby(['BRAND', 'year', 'fuel', 'transmission', 'PRICE_CAT'])[col].transform('mean'),
                   inplace=True)

df['seats'].fillna(df.groupby(['BRAND', 'year', 'fuel', 'transmission', 'PRICE_CAT'])['seats'].transform('median'),
                   inplace=True)

df.dropna(inplace=True)

df['ENGINE_POWER_RATIO'] = df['ENGINE'] / df['MAX_POWER']
df['FUEL_EFF_POWER'] = df['MILEAGE'] / df['MAX_POWER']
df['Power_per_Liter'] = df['MAX_POWER'] / df['ENGINE']
df['Fuel_Efficiency_to_Power'] = df['MILEAGE'] / df['MAX_POWER']
df['Power_per_RPM'] = df['MAX_POWER'] / df['RPM']

result = eda.grab_col_names(df, cat_th=15)
cat_cols, num_cols = result[0], result[1]

eda.rare_analyser(df, target='selling_price', cat_cols=cat_cols)

df.drop(df[df['owner'] == 'Test Drive Car'].index, inplace=True)

df['seats'] = df['seats'].astype(str)

df[['owner', 'fuel', 'seats']] = feng.rare_encoder(df[['owner', 'fuel', 'seats']], 0.07)

feng.lof(df, plot=True)

lof_indexes = feng.lof_indexes(df, 4)

df.drop(lof_indexes, inplace=True)
df.reset_index(inplace=True, drop=True)

df.drop('PRICE_CAT', axis=1, inplace=True)
result = eda.grab_col_names(df, cat_th=15)
cat_cols, num_cols = result[0], result[1]

df = feng.one_hot_encoder(df, cat_cols + ['BRAND'], drop_first=True)

df['selling_price'] = df['selling_price'] * 0.0111

y = df["selling_price"]
X = df.drop(["selling_price"], axis=1)


def base_models(X, y):
    print("Base Models....")
    classifiers = [('LR', LinearRegression()),
                   ('KNN', KNeighborsRegressor()),
                   ('CART', DecisionTreeRegressor()),
                   ('RF', RandomForestRegressor()),
                   ('GBM', GradientBoostingRegressor()),
                   ("XGBoost", XGBRegressor(objective='reg:squarederror')),
                   ("LightGBM", LGBMRegressor()),
                   ("CatBoost", CatBoostRegressor(verbose=False))]
    score = pd.DataFrame(index=['rmse', 'r2_score'])
    for name, classifier in classifiers:
        rmse = np.mean(np.sqrt(-cross_val_score(classifier, X, y, cv=3, scoring="neg_mean_squared_error")))
        r2 = np.mean(cross_val_score(classifier, X, y, cv=3, scoring="r2"))
        score[name] = [rmse, r2]
        print(f'{name} hesaplandı...')
    print(score.T)


rf_model = CatBoostRegressor(random_state=42, max_depth=2, learning_rate=0.1, verbose=False).fit(X, y)
np.mean(np.sqrt(-cross_val_score(rf_model, X, y, cv=3, scoring="neg_mean_squared_error")))
np.mean(cross_val_score(rf_model, X, y, cv=3, scoring="r2"))

eda.plot_importance(rf_model, X)

eda.val_curve_params(rf_model, X, y, "max_depth", range(1, 11), scoring="neg_mean_squared_error")
