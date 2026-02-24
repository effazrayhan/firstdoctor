from thefuzz import process
import json

CONFIDENCE_THRESHOLD = 70

def normalize(text):
    return text.strip().lower()

def load_bundles(path="test_bundles.json"):
    with open(path, "r") as f:
        return json.load(f)

def build_search_index(bundles):
    search_index = {}

    for disease, details in bundles.items():
        search_index[disease] = disease

        for synonym in details.get("synonyms", []):
            search_index[synonym] = disease

    return search_index

def get_tests_for_disease(predicted_disease, threshold=CONFIDENCE_THRESHOLD):
    bundles = load_bundles()
    search_index = build_search_index(bundles)

    normalized_input = normalize(predicted_disease)
    search_terms = list(search_index.keys())

    best_match, score = process.extractOne(normalized_input, search_terms)

    if score >= threshold:
        canonical_disease = search_index[best_match]
        details = bundles[canonical_disease]

        print(f"\nMatch Found: {canonical_disease}")
        print(f"Confidence: {score}%")

        print("Tests:", details.get("tests", []))
        print("Possible Conditions:", details.get("possible_conditions", []))

        return {
            "matched_disease": canonical_disease,
            "confidence": score,
            "tests": details.get("tests", []),
            "possible_conditions": details.get("possible_conditions", [])
        }

    print("\nNo confident match found.")
    return {
        "matched_disease": None,
        "confidence": score,
        "tests": [],
        "possible_conditions": []
    }