from http.server import BaseHTTPRequestHandler
import json
import pickle
import pandas as pd
import os

# ── Load model and columns at cold-start ──
dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(dir_path, 'columns.pkl'), 'rb') as f:
    X_columns = pickle.load(f)


class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            if not data:
                self._json_response({'error': 'No data received'}, 400)
                return

            user_df = pd.DataFrame([data])
            user_df = pd.get_dummies(user_df)
            user_df = user_df.reindex(columns=X_columns, fill_value=0)

            probability = model.predict_proba(user_df)[0][1]
            prediction = int(model.predict(user_df)[0])

            self._json_response({
                'probability': round(probability * 100, 2),
                'prediction': prediction
            })

        except Exception as e:
            self._json_response({'error': str(e)}, 500)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
