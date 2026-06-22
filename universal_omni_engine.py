import os
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class UniversalOmniEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        self.raw_records = []
        self.record_embeddings = None
        self.user_profiles = {}

    def _flatten_unstructured_record(self, data):
        """Recursively treats all unknown keys, values, and tags as semantic text signals."""
        tokens = []
        if isinstance(data, dict):
            for key, val in data.items():
                tokens.append(str(key).replace('_', ' ').replace('-', ' '))
                tokens.extend(self._flatten_unstructured_record(val))
        elif isinstance(data, list):
            for item in data:
                tokens.extend(self._flatten_unstructured_record(item))
        else:
            tokens.append(str(data))
        return tokens

    def train_and_index(self, domain_json_path):
        """Loads a variable dataset, flattens items dynamically, and saves the vector space."""
        with open(domain_json_path, 'r') as f:
            self.raw_records = json.load(f)
            
        flat_texts = []
        for record in self.raw_records:
            tokens = self._flatten_unstructured_record(record)
            flat_texts.append(" ".join(tokens))
            
        self.record_embeddings = self.vectorizer.fit_transform(flat_texts)
        print(f"Index Built: Learned {len(self.raw_records)} irregular records from {domain_json_path}.")

    def _inject_scores_into_unknown_hierarchy(self, node, score):
        """Recursively walks the item's original unconstrained hierarchy and injects scores."""
        if isinstance(node, dict):
            new_dict = {}
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    # Branch layer: key maps to its calculated score and its deep elements
                    new_dict[k] = [score, self._inject_scores_into_unknown_hierarchy(v, score)]
                else:
                    # Leaf layer: key maps directly to the score float
                    new_dict[k] = score
            return new_dict
        elif isinstance(node, list):
            # Array layer: map scores straight to tag array elements
            return [{str(item): score} if not isinstance(item, (dict, list)) else self._inject_scores_into_unknown_hierarchy(item, score) for item in node]
        return node

    def search(self, query_text, top_n=1):
        """Phase 1: Search. Returns the matching record's unique JSON layout with scores."""
        if self.record_embeddings is None: return []
        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.record_embeddings).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            score_val = round(float(similarities[idx]), 4)
            if score_val > 0.0:
                scored_json = self._inject_scores_into_unknown_hierarchy(self.raw_records[idx], score_val)
                results.append(scored_json)
        return results

    def recommend(self, record_index, top_n=1):
        """Phase 2: Recommendation. Evaluates an item's unique layout against all others in the file."""
        if self.record_embeddings is None or record_index >= len(self.raw_records): return []
        
        target_vec = self.record_embeddings[record_index]
        similarities = cosine_similarity(target_vec, self.record_embeddings).flatten()
        similarities[record_index] = -1  # Exclude self
        
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            score_val = round(float(similarities[idx]), 4)
            if score_val > 0.0:
                scored_json = self._inject_scores_into_unknown_hierarchy(self.raw_records[idx], score_val)
                results.append(scored_json)
        return results

    def log_interaction(self, user_id, record_index):
        """Feedback Loop: Saves user activity signals to continuously adjust profile weights."""
        if self.record_embeddings is None or record_index >= len(self.raw_records): return
        
        action_vec = self.record_embeddings[record_index].toarray().flatten()
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = action_vec
        else:
            self.user_profiles[user_id] = (self.user_profiles[user_id] + action_vec) / 2.0

    def personalize_search(self, user_id, query_text, top_n=1):
        """Phase 3: Personalization. Adjusts search results based on user click history."""
        if self.record_embeddings is None: return []
        query_vec = self.vectorizer.transform([query_text]).toarray().flatten()
        
        if user_id in self.user_profiles:
            user_profile = self.user_profiles[user_id]
            # Blend: 60% text query string + 40% history profile affinity values
            blended_vec = (query_vec * 0.6) + (user_profile * 0.4)
        else:
            blended_vec = query_vec
            
        blended_vec = blended_vec.reshape(1, -1)
        similarities = cosine_similarity(blended_vec, self.record_embeddings).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            score_val = round(float(similarities[idx]), 4)
            if score_val > 0.0:
                scored_json = self._inject_scores_into_unknown_hierarchy(self.raw_records[idx], score_val)
                results.append(scored_json)
        return results

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"-> Model binary successfully saved to: {filepath}")

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
