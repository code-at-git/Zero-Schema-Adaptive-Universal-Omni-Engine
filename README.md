# Zero-Schema Adaptive Universal Omni-Engine
### Architecture Framework for Search, Recommendation, and Context Personalization Mappings

Hey there! If you’ve ever tried to build a search, recommendation, or personalization engine using standard machine learning tools, you know how incredibly frustrating it is when your data doesn't have a rigid structure. The moment a category, key, or tag list changes from one item to the next, traditional classifiers fall flat.

This project solves that exact problem. It is an adaptive, zero-schema matching and context routing system. It doesn't care about field names, nesting depths, or taxonomy constraints. You can feed it a movie catalog, an e-commerce product feed, a digital image repository, or tech support logs—each with entirely irregular columns and uneven depths—and it will instantly adapt.

The engine operates across all fields, calculates similarities, maps user behavior contexts on the fly, and outputs hierarchically unstructured label-to-score JSON payloads that perfectly match the original data shape without using generic, hardcoded structural keys.

---

## 💡 Core Philosophy & Techniques

Instead of fighting unstructured data with strict classification trees, this system leans into a couple of really elegant machine learning and structural data concepts:

### 1. The "Bag of Semantic Signals"
The engine treats everything inside a single JSON record—the keys, the sub-keys, the terminal values, and the tag array items—as an unconstrained text footprint. A recursive flattening algorithm strips away all syntax tokens and converts the entire item block into a continuous text representation.

### 2. Multi-Gram Vector Space Modeling
Behind the scenes, we use Scikit-Learn's `TfidfVectorizer` configured with an `ngram_range=(1, 3)`.

* **Why this matters:** It doesn't just look at isolated words. It captures phrases up to three words long (like *Active Noise Cancelling* or *Pastel Pink Hoodie*).
* **Global Weighting:** The system weights these terms globally across the domain using *Inverse Document Frequency*. Rare, highly descriptive words automatically float to the top of the relevance calculations, while common formatting text is down-weighted.

### 3. Cosine Similarity Mapping
We use `cosine_similarity` to measure the angular alignment between query vectors, item text blocks, and user context history profiles. This enables lightning-fast calculations without needing heavy, resource-intensive neural networks.

### 4. Recursive Score Injection
When generating predictions, the system maps the raw item data shape dynamically in memory.

* **Branch Layers:** Formatted into a simple data pair wrapper array: `[Score, Nested_Object]`.
* **Leaf Elements:** Map your custom taxonomy labels directly to the computed float value: `"Tag_Name": 0.881`.
* **Zero Wrappers:** It completely eliminates rigid hardcoded text wrappers like `"score"`, `"label"`, or `"child"`.

---

## 🛠️ The Three Operational Modes

To make this server-ready, the workflow is split into three decoupled operational states so you aren't constantly reading raw data files during live requests.

### Mode 1: Initial Training (TRAIN)
* **What it does:** Reads a raw dataset JSON file from disk once, learns its entire vocabulary structure, computes the baseline vector arrays, and serializes the complete instance to disk as a `.pkl` model binary file.
* **Why it's cool:** It isolates your domains. A movies domain and a support tickets domain create entirely separate model states.

### Mode 2: Live Inferences (PREDICT)
* **What it does:** Loads a pre-compiled `.pkl` binary directly from your drive to serve immediate requests. It completely ignores and skips the original JSON data files during this step.
* **The Actions:**
  * **Search:** Matches a standard keyword phrase against the domain footprint.
  * **Recommend:** Passes an item's array index to find adjacent items with the tightest feature overlap.
  * **Personalize:** Evaluates an incoming search query but biases the results based on an internal running-average profile cached for that specific user.

### Mode 3: Retraining & Adaptation (RETRAIN)
* **What it does:** Acts as the continuous feedback pipeline. When you get new data items or user-corrected inputs, the framework appends the new unstructured schema straight to your data source file and re-runs the indexing routine to update the disk weights automatically.

---

## 📁 System Architecture & Directory Layout

The app is neatly structured to separate your data inputs, serialized models, front-end layers, and historical logs:  

```text
your_project/
│
├── universal_omni_engine.py   # Core ML algorithms, flattening, and score injectors
├── run_system.py              # CLI mode runner for terminal executions
├── app.py                     # Flask web server exposing REST endpoints
│
├── dataset/                   # Folder containing your unstructured JSON files
│   ├── domain_movies.json     
│   └── domain_products.json   
│
├── models/                    # Folder storing optimized .pkl checkpoint binaries
│   └── movies_omni.pkl        
│
├── saved_predictions/         # Folder storing archived prediction snapshots
│
└── templates/
    └── index.html             # The unified four-block dashboard interface UI
```

---

## 🚀 How To Run & Use the System

### Workspace Setup
First, make sure you have the required machine learning and web dependencies installed via your terminal:

```bash
pip install flask numpy scikit-learn
```

### Running the Web UI Interface
To control everything through an interactive web browser dashboard, launch the Flask web router:

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`.

From the dashboard interface, you can click on any dataset to browse raw records, spin up a new model, run searches, view live dynamic prediction responses, click check-boxes to log archived snapshots, and append new records to fine-tune the system.

<img src="/Zero-Schema-Adaptive-Omni-Engine-Dashboard.png" alt="Web UI Dashboard" width="600">

### Running via Command Line (CLI)
If you prefer script-based pipelines or background automations, execute your operational modes directly using the orchestrator file:

```bash
python run_system.py
```

---

## 📏 Rules for Adding Custom JSON Files

You can drop any valid JSON file into the `dataset/` directory and it will work out of the box, as long as it respects these three universal data formatting rules:

1. **The Root Layer Must Be an Array:** The system needs a list of items to evaluate. Your file must start and end with square brackets `[...]` containing independent JSON objects.
2. **No Raw Math Matrices:** Keep values restricted to strings, booleans, integers, or arrays of words. Dumping flat arrays of image pixel values or map coordinates will clutter the vectorizer space.
3. **Use Descriptors for Combined Concepts:** If a specific phrase belongs together as a unified tag, group it with an underscore or dash (e.g., use `"Space_Travel"` or `"Hi-Res_Audio"` instead of `"Space Travel"`). This tells the tokenizer to treat it as an exact token match.
