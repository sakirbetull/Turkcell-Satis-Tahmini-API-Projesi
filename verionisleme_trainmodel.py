import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize

df = pd.read_csv(r'C:\GYK\project\tabledf.csv')

class verionisleme:
    @staticmethod
    def missing_data_manipulation(df):
        # print(df.isnull().sum())
        df.fillna(0, inplace=True)  # Tüm eksik değerleri 0 ile doldurur
        return df

    @staticmethod
    def winsorize_unit_price(df):
        df['winsorize_unit_price'] = pd.Series(winsorize(df['unit_price'].values, limits=[0.05, 0.05]))
        return df

    @staticmethod
    def total_price_calculation(df):
        df['total_price'] = ((1 - df['discount']) * (df['unit_price'] * df['quantity']))
        return df

    @staticmethod
    def order_date_manipulation(df):
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        return df

    @staticmethod
    def year_quarter_calculation(df):
        df['Year'] = df['order_date'].dt.year
        df['yearquarter'] = df['Year'].astype(str) + ' Q' + df['order_date'].dt.quarter.astype(str)
        return df

    @staticmethod
    def discount_effective_calculation(df):
        df = df.sort_values(by=['product_id', 'order_date'])
        df['discount_effective'] = ((df['discount'] > 0) & 
                                    (df['quantity'] > df.groupby(['product_id', 'yearquarter'])['quantity'].shift(1))
                                   ).astype(int)
        return df

def preprocess_data(df):
    # Veri ön işleme adımları
    df = verionisleme.missing_data_manipulation(df)
    df = verionisleme.winsorize_unit_price(df)
    df = verionisleme.total_price_calculation(df)
    df = verionisleme.order_date_manipulation(df)
    df = verionisleme.year_quarter_calculation(df)
    df = verionisleme.discount_effective_calculation(df)

    # Kategorik değişkenleri one-hot encoding ile dönüştür
    categorical_columns = ['product_name', 'category_name', 'yearquarter', 'city']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    return df

def train_and_save_model():
    # Veri setini yükleyin
    df = pd.read_csv(r'C:\GYK\project\tabledf.csv')

    # Veri ön işleme
    df = preprocess_data(df)

    # Özellikler ve hedef değişkeni ayırma
    X = df.drop(columns=['discount_effective', 'unit_price', 'order_date', 'customer_id', 'product_id', 'units_in_stock', 'category_id', 'Year'])
    y = df['discount_effective']

    # SMOTE ile veri dengeleme
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    # Hiperparametre optimizasyonu
    param_grid = {
        "max_depth": [3, 5, 8, 10],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 5, 10]
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_balanced, y_balanced)

    # En iyi modeli kaydet
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'salesprediction_decisiontree_model.pkl')
    print("Model başarıyla kaydedildi.")

if __name__ == "__main__":
    train_and_save_model()

