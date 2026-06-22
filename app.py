import os
import json
import uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for
from universal_omni_engine import UniversalOmniEngine

app = Flask(__name__)

# System Directory Configurations
DATASET_DIR = "dataset"
MODEL_DIR = "models"
PRED_DIR = "saved_predictions"

for folder in [DATASET_DIR, MODEL_DIR, PRED_DIR]:
    os.makedirs(folder, exist_ok=True)

# Helper function to get tracked file states
def get_system_inventories():
    datasets = [f for f in os.listdir(DATASET_DIR) if f.endswith('.json')]
    models = [f.replace('.pkl', '') for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
    predictions = [f for f in os.listdir(PRED_DIR) if f.endswith('.json')]
    return datasets, models, predictions

@app.route('/')
def index():
    datasets, models, predictions = get_system_inventories()
    return render_template('index.html', datasets=datasets, models=models, predictions=predictions)

# =====================================================================
# DATASET MANAGEMENT ENDPOINTS
# =====================================================================
@app.route('/dataset/create', methods=['POST'])
def create_dataset():
    filename = request.form.get('filename')
    if not filename.endswith('.json'):
        filename += '.json'
    
    path = os.path.join(DATASET_DIR, filename)
    # Write a clean, valid starting empty JSON array
    with open(path, 'w') as f:
        json.dump([], f)
    return redirect(url_for('index'))

@app.route('/dataset/view/<filename>')
def view_dataset(filename):
    path = os.path.join(DATASET_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Dataset not found"}), 404

@app.route('/dataset/add_record', methods=['POST'])
def add_record():
    filename = request.form.get('dataset_name')
    raw_json_str = request.form.get('json_record')
    
    path = os.path.join(DATASET_DIR, filename)
    try:
        new_record = json.loads(raw_json_str)
        with open(path, 'r') as f:
            current_data = json.load(f)
            
        current_data.append(new_record)
        with open(path, 'w') as f:
            json.dump(current_data, f, indent=2)
        return redirect(url_for('index'))
    except Exception as e:
        return f"Invalid JSON string error: {str(e)}", 400

# =====================================================================
# TRAINING & RETRAINING ENDPOINT (Mode 1 & Mode 3)
# =====================================================================
@app.route('/model/train', methods=['POST'])
def train_model():
    dataset_name = request.form.get('dataset_name')
    model_name = request.form.get('model_name')
    
    dataset_path = os.path.join(DATASET_DIR, dataset_name)
    engine = UniversalOmniEngine()
    engine.train_and_index(dataset_path)
    
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    engine.save(model_path)
    return redirect(url_for('index'))

# =====================================================================
# CORE ENGINE INFERENCE ENDPOINT (Mode 2: Live Predict)
# =====================================================================
@app.route('/model/predict', methods=['POST'])
def predict():
    model_name = request.form.get('model_name')
    mode = request.form.get('mode') # search, recommend, or personalize
    query = request.form.get('query', '')
    interact_idx = request.form.get('interact_idx', '')
    user_id = request.form.get('user_id', 'GLOBAL_USER')
    save_flag = request.form.get('save_prediction') == 'true'
    
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        return jsonify({"error": f"Model artifact '{model_name}' missing from drive."}), 404
        
    # Dynamically extract and load our locked-in model binary state
    pipeline = UniversalOmniEngine.load(model_path)
    result_output = []
    
    if mode == "search":
        result_output = pipeline.search(query, top_n=3)
        
    elif mode == "recommend":
        try:
            idx = int(interact_idx)
            result_output = pipeline.recommend(record_index=idx, top_n=3)
        except ValueError:
            return jsonify({"error": "Recommendation mode requires an integer record index."}), 400
            
    elif mode == "personalize":
        try:
            idx = int(interact_idx)
            # 1. Log structural user interest feedback
            pipeline.log_interaction(user_id, record_index=idx)
            # 2. Re-save model layer to retain the internal adaptive user profile vector weights
            pipeline.save(model_path)
            # 3. Output context-shifted prediction array
            result_output = pipeline.personalize_search(user_id, query, top_n=3)
        except ValueError:
            return jsonify({"error": "Personalization mode requires a valid feedback integer record index."}), 400

    # Handle explicit browser request to archive this prediction
    if save_flag:
        pred_filename = f"pred_{model_name}_{uuid.uuid4().hex[:6]}.json"
        pred_path = os.path.join(PRED_DIR, pred_filename)
        log_payload = {
            "metadata": {"model": model_name, "operation_mode": mode, "query": query, "user": user_id},
            "output_tree": result_output
        }
        with open(pred_path, 'w') as f:
            json.dump(log_payload, f, indent=2)

    return jsonify(result_output)

@app.route('/prediction/view/<filename>')
def view_prediction(filename):
    path = os.path.join(PRED_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Saved log profile missing"}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
