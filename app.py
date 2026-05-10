from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)

# ── Load model and columns ──
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('columns.pkl', 'rb') as f:
    X_columns = pickle.load(f)

print("Model loaded successfully!")
print(f"Expecting {len(X_columns)} columns")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data received'}), 400

        user_df = pd.DataFrame([data])
        user_df = pd.get_dummies(user_df)
        user_df = user_df.reindex(columns=X_columns, fill_value=0)

        probability = model.predict_proba(user_df)[0][1]
        prediction  = int(model.predict(user_df)[0])

        return jsonify({
            'probability': round(probability * 100, 2),
            'prediction':  prediction
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)