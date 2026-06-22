import os
import json
from universal_omni_engine import UniversalOmniEngine

# Core directory configurations
MODEL_DIR = "models"
DATASET_DIR = "dataset"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True) # Automatically creates the dataset directory if missing

# =====================================================================
# OPERATIONAL MODE 1: INITIAL TRAINING
# =====================================================================
def run_train_mode(json_file_name, model_name):
    """Mode: Train.

    Reads the raw unstructured JSON file from the dataset folder,
    trains the model, and saves it to disk.
    """
    json_path = os.path.join(DATASET_DIR, json_file_name)
    print(f"\n==============================================================")
    print(f"[MODE: INITIAL TRAIN] Building Model for File: {json_path}")
    print(f"==============================================================")
    
    if not os.path.exists(json_path):
        print(f"[ERROR] Dataset file {json_path} missing. Please move your JSON file to the '{DATASET_DIR}/' folder.")
        return None
        
    engine = UniversalOmniEngine()
    engine.train_and_index(json_path)
    
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    engine.save(model_path)
    return model_path

# =====================================================================
# OPERATIONAL MODE 2: LIVE PREDICTIONS (Zero JSON Parsing)
# =====================================================================
def run_predict_mode(model_name, query, interact_idx, ambiguous_query, user_id):
    """Mode: Predict.

    Loads the saved model directly from disk to run search,
    recommendations, and personalization. It completely ignores the
    original JSON files.
    """
    print(f"\n==============================================================")
    print(f"[MODE: PREDICT] Serving Inferences from Stored Model: '{model_name}'")
    print(f"==============================================================")
    
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        print(f"[ERROR] Stored binary {model_path} missing. Please run Train Mode first.")
        return
        
    pipeline = UniversalOmniEngine.load(model_path)
    
    # Phase 1: Search Prediction
    print(f"\n[1. SEARCH RESULT] Query: '{query}'")
    search_out = pipeline.search(query, top_n=1)
    print(json.dumps(search_out, indent=4))
    
    # Phase 2: Recommendation Generation
    print(f"\n[2. RECOMMENDATION] Structural matches for record index {interact_idx}:")
    rec_out = pipeline.recommend(record_index=interact_idx, top_n=1)
    print(json.dumps(rec_out, indent=4))
    
    # Phase 3: Log Interaction Feedback
    print(f"\n[3. FEEDBACK LOGGED]")
    print(f"User '{user_id}' interacts with record index {interact_idx}. Adapting context profile...")
    pipeline.log_interaction(user_id, record_index=interact_idx)
    
    # Phase 4: Personalized Search
    print(f"\n[4. PERSONALIZED SEARCH SELECTION] Query: '{ambiguous_query}'")
    standard_search = pipeline.search(ambiguous_query, top_n=1)
    personalized_search = pipeline.personalize_search(user_id, ambiguous_query, top_n=1)
    
    print("Standard Result (No Profile Context):")
    print(json.dumps(standard_search, indent=4))
    print(f"Personalized Result (Biased by User '{user_id}' history):")
    print(json.dumps(personalized_search, indent=4))
    
    # Re-save to disk to store the updated user personalization history vectors
    pipeline.save(model_path)

# =====================================================================
# OPERATIONAL MODE 3: RETRAIN / ADAPT (Feedback Data Ingestion)
# =====================================================================
def run_retrain_mode(json_file_name, model_name, new_unstructured_feedback_list):
    """Mode: Retrain.

    Appends new data items or user-corrected schemas to the domain file
    inside the dataset directory, then triggers a fast re-index to
    update the weights on disk.
    """
    json_path = os.path.join(DATASET_DIR, json_file_name)
    print(f"\n==============================================================")
    print(f"[MODE: RETRAIN/ADAPT] Appending New Feedback to: {json_path}")
    print(f"==============================================================")
    
    # 1. Read existing data array from dataset directory
    with open(json_path, 'r') as f:
        current_catalog = json.load(f)
        
    # 2. Add new unstructured records to perfect the weights
    for new_item in new_unstructured_feedback_list:
        current_catalog.append(new_item)
        
    # 3. Write back the updated file to the dataset directory
    with open(json_path, 'w') as f:
        json.dump(current_catalog, f, indent=2)
        
    # 4. Trigger automatic re-training to sync the changes with your disk file
    run_train_mode(json_file_name, model_name)


# =====================================================================
# EXECUTING RUNTIME SEPARATION THROUGH ALL 5 DOMAINS
# =====================================================================
if __name__ == "__main__":
    
    # -----------------------------------------------------------------
    # STEP 1: INITIAL TRAINING PASS (Looks directly inside dataset/)
    # -----------------------------------------------------------------
    run_train_mode("domain_movies.json", "movies_omni")
    run_train_mode("domain_tickets.json", "tickets_omni")
    run_train_mode("domain_products.json", "products_omni")
    run_train_mode("domain_images.json", "images_omni")
    run_train_mode("domain_videos.json", "videos_omni")
    
    # -----------------------------------------------------------------
    # STEP 2: LIVE INFERENCE RUNS (Fast serving without reading JSON)
    # -----------------------------------------------------------------
    run_predict_mode(
        model_name="movies_omni",
        query="space travel exploration movies by nolan",
        interact_idx=1,
        ambiguous_query="Nolan movies",
        user_id="MOVIE_USER_88"
    )
    
    run_predict_mode(
        model_name="tickets_omni",
        query="database timeout connection error on production pool",
        interact_idx=0,
        ambiguous_query="production pool issues",
        user_id="DBA_USER_99"
    )

    run_predict_mode(
        model_name="products_omni",
        query="active noise cancelling wire free audio gear",
        interact_idx=1,
        ambiguous_query="high performance gear",
        user_id="USER_ALPHA_PROD"
    )
    
    run_predict_mode(
        model_name="images_omni",
        query="cyberpunk neon pink tokyo wide angle photography",
        interact_idx=0,
        ambiguous_query="travel photography",
        user_id="USER_BETA_IMG"
    )

    run_predict_mode(
        model_name="videos_omni",
        query="python engineering coding tutorials for developers",
        interact_idx=1,
        ambiguous_query="software developer life",
        user_id="USER_GAMMA_VID"
    )

    # -----------------------------------------------------------------
    # STEP 3: RETRAIN DEMO (Saves updates inside dataset/ folder)
    # -----------------------------------------------------------------
    new_unstructured_video_correction = [
        {
            "clip_id": "V-FEEDBACK-99",
            "title_caption": "Advanced pandas data analytics scripts for software developers and data engineers",
            "nested_metrics": { "views": 5000 },
            "random_tags_list": ["Python", "Data_Science"]
        }
    ]
    
    # Ingest the new data and automatically re-index the video pkl file
    run_retrain_mode(
        json_file_name="domain_videos.json",
        model_name="videos_omni",
        new_unstructured_feedback_list=new_unstructured_video_correction
    )
    
    # Instantly verify the prediction mode works with the updated data weights
    run_predict_mode(
        model_name="videos_omni",
        query="pandas data analytics scripts",
        interact_idx=2,
        ambiguous_query="data science",
        user_id="USER_GAMMA_VID"
    )
