import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from verionisleme_trainmodel import preprocess_data  # Veri işleme fonksiyonlarını buradan alıyoruz

app = FastAPI(title='Sales Prediction API', description='Sales Prediction API using Decision Trees Model')

class Applicant(BaseModel):
    order_date: str
    customer_id: str
    city: str
    product_id: int
    product_name: str
    units_in_stock: int
    unit_price: float
    quantity: int
    discount: float
    category_id: int
    category_name: str

# Modeli yükle
try:
    model = joblib.load('C:\GYK\project\salesprediction_decisiontree_model.pkl')
except FileNotFoundError:
    raise RuntimeError("Model dosyası bulunamadı. Lütfen 'salesprediction_decisiontree_model.pkl' dosyasını kontrol edin.")

@app.post('/predict', tags=['prediction'])
def predict_approval(applicant: Applicant):
    # API'ye gelen ham veriyi DataFrame'e dönüştür
    input_data = pd.DataFrame([{
        'order_date': applicant.order_date,
        'customer_id': applicant.customer_id,
        'city': applicant.city,
        'product_id': applicant.product_id,
        'product_name': applicant.product_name,
        'units_in_stock': applicant.units_in_stock,
        'unit_price': applicant.unit_price,
        'quantity': applicant.quantity,
        'discount': applicant.discount,
        'category_id': applicant.category_id,
        'category_name': applicant.category_name
    }])

    # Veri ön işleme
    input_data = preprocess_data(input_data)

    # Modelin beklediği sütunları seç
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Tahmin yap
    prediction = model.predict(input_data)[0]
    result = 'Increased' if prediction == 1 else 'Not Increased'

    return {
        'prediction': result,
        'details': input_data.iloc[0].to_dict()
    }
