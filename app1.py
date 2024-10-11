from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

# Tải mô hình đã được lưu
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form
    try:
        fixed_acidity = float(request.form['fixed_acidity'])
        volatile_acidity = float(request.form['volatile_acidity'])
        citric_acid = float(request.form['citric_acid'])
        residual_sugar = float(request.form['residual_sugar'])
        chlorides = float(request.form['chlorides'])
        free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
        total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
        density = float(request.form['density'])
        ph = float(request.form['pH'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol'])
        
        # Tạo array từ dữ liệu người dùng nhập vào
        input_features = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                    free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]])

        # Dự đoán kết quả
        prediction = model.predict(input_features)
        
        # Trả về kết quả dự đoán
        return render_template('index1.html', prediction_text=f"Chất lượng rượu dự đoán: {int(prediction[0])}")
    
    except Exception as e:
        return render_template('index1.html', prediction_text="Lỗi khi dự đoán. Vui lòng nhập đúng dữ liệu.")

if __name__ == '__main__':
    app.run(debug=True)
