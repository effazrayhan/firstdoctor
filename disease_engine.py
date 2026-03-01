"""
disease_engine.py — Unified Disease Detection Pipeline

Merges spaCy keyword extraction, PyTorch model prediction,
symptom/test mapping, and PDF prescription generation into
a single DiseaseDetector class.

Usage:
    from disease_engine import DiseaseDetector

    detector = DiseaseDetector()
    result   = detector.run("I have a headache and fever for 3 days")
    detector.generate_pdf(result, patient_name="Jane Doe")
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spacy
import torch
from fpdf import FPDF
from sklearn.preprocessing import LabelEncoder
from thefuzz import process

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _BASE_DIR / "model"
_DATASET_PATH = _MODEL_DIR / "database" / "dataset.csv"
_WEIGHTS_PATH = _MODEL_DIR / "torch_symptom_model.pth"
_TEST_BUNDLES_PATH = _BASE_DIR / "test_bundles.json"
_FUZZY_THRESHOLD = 70


# ---------------------------------------------------------------------------
# PyTorch model definition (mirrors model/modelrun.py)
# ---------------------------------------------------------------------------
class SymptomClassifier(torch.nn.Module):
    """Two-layer feed-forward classifier: symptoms → disease logits."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class DiseaseDetector:
    """End-to-end disease detection: text → symptoms → prediction → PDF."""

    def __init__(
        self,
        dataset_path: str | Path = _DATASET_PATH,
        weights_path: str | Path = _WEIGHTS_PATH,
        test_bundles_path: str | Path = _TEST_BUNDLES_PATH,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.weights_path = Path(weights_path)
        self.test_bundles_path = Path(test_bundles_path)

        # ---- Phase 1 setup: spaCy NLP pipeline ----
        self.nlp = self._load_spacy(spacy_model)

        # ---- Phase 2 setup: dataset, encoder, model ----
        self.df: pd.DataFrame = self._load_dataset()
        self.symptom_names: list[str] = self._extract_symptom_names()
        self.label_encoder: LabelEncoder = self._fit_label_encoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: SymptomClassifier = self._load_model()

        # ---- Phase 3 setup: test bundles ----
        self.test_bundles: dict = self._load_test_bundles()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_spacy(model_name: str) -> spacy.language.Language:
        """Load a spaCy model.  Falls back to a blank English model if the
        requested model is not installed, and adds a simple entity ruler
        so that common symptom phrases are recognised."""
        try:
            nlp = spacy.load(model_name)
        except OSError:
            print(
                f"[DiseaseDetector] spaCy model '{model_name}' not found. "
                "Using blank 'en' model.  Run:  python -m spacy download "
                f"{model_name}"
            )
            nlp = spacy.blank("en")
        return nlp

    def _load_dataset(self) -> pd.DataFrame:
        """Load the training CSV.  The dataset is required at runtime so
        that we can reconstruct the LabelEncoder and know the symptom
        column order expected by the model."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}.  "
                "If the file is stored in Git LFS, run:  "
                "git lfs pull --include='model/database/dataset.csv'"
            )
        df = pd.read_csv(self.dataset_path)
        # Guard against LFS pointer files
        if df.shape[1] < 2:
            raise ValueError(
                f"Dataset at {self.dataset_path} appears to be a Git LFS "
                "pointer (only {df.shape[1]} column(s)).  Pull the real "
                "file with:  git lfs pull"
            )
        return df

    def _extract_symptom_names(self) -> list[str]:
        """Return the ordered list of symptom column names (everything
        except the 'diseases' label column)."""
        cols = [c for c in self.df.columns if c.lower() != "diseases"]
        return cols

    def _fit_label_encoder(self) -> LabelEncoder:
        le = LabelEncoder()
        le.fit(self.df["diseases"])
        return le

    def _load_model(self) -> SymptomClassifier:
        input_dim = len(self.symptom_names)
        output_dim = len(self.label_encoder.classes_)
        model = SymptomClassifier(input_dim, output_dim)
        state_dict = torch.load(self.weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _load_test_bundles(self) -> dict:
        if not self.test_bundles_path.exists():
            print(
                f"[DiseaseDetector] test_bundles.json not found at "
                f"{self.test_bundles_path}.  Test mapping will be empty."
            )
            return {}
        with open(self.test_bundles_path, "r") as f:
            return json.load(f)

    def _normalise_symptom(self, sym: str) -> str:
        """Normalise a symptom column name to a lowercase phrase.
        Handles both underscore-separated (``chest_pain``) and
        space-separated (``chest pain``) column names."""
        return sym.replace("_", " ").strip().lower()

    def _ensure_symptom_ruler(self) -> None:
        """Add an EntityRuler to the spaCy pipeline (once) that tags
        symptom column names as SYMPTOM entities.  This bridges the gap
        left by the original medspacy.py snippet."""
        if "symptom_ruler" in self.nlp.pipe_names:
            return
        ruler = self.nlp.add_pipe("entity_ruler", name="symptom_ruler")
        patterns = []
        for sym in self.symptom_names:
            phrase = self._normalise_symptom(sym)
            if phrase:
                patterns.append({"label": "SYMPTOM", "pattern": phrase})
        ruler.add_patterns(patterns)

    # ------------------------------------------------------------------
    # Phase 1 — spaCy keyword / symptom extraction
    # ------------------------------------------------------------------
    def extract_symptoms(self, text: str) -> list[str]:
        """Extract symptom keywords from free-text input using spaCy
        entity recognition plus simple substring matching against known
        symptom column names.

        Returns a list of matched symptom column names (underscore form).
        """
        self._ensure_symptom_ruler()

        text_lower = text.lower()
        doc = self.nlp(text_lower)

        matched: set[str] = set()

        # 1. Entities recognised by spaCy / entity ruler
        for ent in doc.ents:
            ent_text = ent.text.strip().lower()
            for sym in self.symptom_names:
                clean = self._normalise_symptom(sym)
                if clean == ent_text or ent_text in clean or clean in ent_text:
                    matched.add(sym)

        # 2. Direct substring scan (catches symptoms the ruler may miss
        #    due to tokenisation differences)
        for sym in self.symptom_names:
            clean = self._normalise_symptom(sym)
            if clean and clean in text_lower:
                matched.add(sym)

        return sorted(matched)

    # ------------------------------------------------------------------
    # Phase 2 — Model prediction
    # ------------------------------------------------------------------
    def predict_diseases(
        self, symptoms: list[str], top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Build a binary symptom vector from *symptoms* (column names)
        and return the top-k predicted diseases with probabilities.

        Each element: {"disease": str, "probability": float}
        """
        # Build binary vector
        vector = np.zeros(len(self.symptom_names), dtype=np.float32)
        for i, sym in enumerate(self.symptom_names):
            if sym in symptoms:
                vector[i] = 1.0

        # Predict
        with torch.no_grad():
            inputs = torch.tensor(vector, dtype=torch.float32).to(self.device)
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=0).cpu().numpy()

        sorted_idx = np.argsort(probs)[::-1][:top_k]
        results = []
        for idx in sorted_idx:
            results.append(
                {
                    "disease": self.label_encoder.classes_[idx],
                    "probability": float(probs[idx]),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Phase 3 — Symptom / test mapping from scraped data
    # ------------------------------------------------------------------
    def get_disease_symptoms(self, disease_name: str) -> list[str]:
        """Look up the dataset for rows matching *disease_name* and
        return the symptom columns that are set to 1 for that disease."""
        mask = self.df["diseases"].str.lower() == disease_name.strip().lower()
        subset = self.df.loc[mask, self.symptom_names]
        if subset.empty:
            return []
        # A symptom is associated if it appears in *any* row for that disease
        active = subset.any(axis=0)
        return [sym for sym, flag in active.items() if flag]

    def get_recommended_tests(self, disease_name: str) -> dict[str, Any]:
        """Use fuzzy matching (thefuzz) against test_bundles.json to find
        recommended lab tests for *disease_name*.

        Returns a dict with keys: matched_disease, confidence, tests,
        possible_conditions, escalation (if applicable).
        """
        result: dict[str, Any] = {
            "matched_disease": None,
            "confidence": 0,
            "tests": [],
            "possible_conditions": [],
            "escalation": None,
        }

        # --- Try diseases section (fuzzy.py style) ---
        diseases_section = self.test_bundles.get("diseases", {})
        top_keys = {k: k for k in diseases_section.keys()}
        # Also index synonyms from diseases section
        for disease, details in diseases_section.items():
            if isinstance(details, dict):
                for syn in details.get("synonyms", []):
                    top_keys[syn.lower()] = disease

        if top_keys:
            search_terms = list(top_keys.keys())
            best_match, score = process.extractOne(
                disease_name.strip().lower(), search_terms
            )
            if score >= _FUZZY_THRESHOLD:
                canonical = top_keys[best_match]
                details = diseases_section[canonical]
                result.update(
                    {
                        "matched_disease": canonical,
                        "confidence": score,
                        "tests": details.get("tests", []),
                        "possible_conditions": details.get(
                            "possible_conditions", []
                        ),
                    }
                )

        # --- Check escalation_logic section ---
        esc_logic = self.test_bundles.get("escalation_logic", {})
        if esc_logic:
            esc_keys = list(esc_logic.keys())
            best_esc, esc_score = process.extractOne(
                disease_name.strip().lower(), esc_keys
            )
            if esc_score >= _FUZZY_THRESHOLD:
                esc_data = esc_logic[best_esc]
                action = esc_data.get("action", {})
                # Merge escalation tests into the result
                esc_tests = action.get("recommend_tests", [])
                result["tests"] = list(
                    set(result["tests"] + esc_tests)
                )
                result["escalation"] = {
                    "rule": best_esc,
                    "priority": action.get("priority", "UNKNOWN"),
                    "notes": action.get("notes", ""),
                }
                if not result["matched_disease"]:
                    result["matched_disease"] = best_esc
                    result["confidence"] = esc_score

        return result

    # ------------------------------------------------------------------
    # Convenience: full pipeline
    # ------------------------------------------------------------------
    def run(self, text: str, top_k: int = 3) -> dict[str, Any]:
        """Execute the full pipeline: extract → predict → map.

        Returns a dict:
            {
                "input_text": str,
                "extracted_symptoms": [str, ...],
                "predictions": [
                    {
                        "disease": str,
                        "probability": float,
                        "known_symptoms": [str, ...],
                        "recommended_tests": {...},
                    },
                    ...
                ],
            }
        """
        symptoms = self.extract_symptoms(text)
        predictions = self.predict_diseases(symptoms, top_k=top_k)

        enriched: list[dict[str, Any]] = []
        for pred in predictions:
            disease = pred["disease"]
            enriched.append(
                {
                    **pred,
                    "known_symptoms": self.get_disease_symptoms(disease),
                    "recommended_tests": self.get_recommended_tests(disease),
                }
            )

        return {
            "input_text": text,
            "extracted_symptoms": symptoms,
            "predictions": enriched,
        }

    # ------------------------------------------------------------------
    # Phase 4 — PDF prescription generation
    # ------------------------------------------------------------------
    def generate_pdf(
        self,
        result: dict[str, Any],
        patient_name: str = "N/A",
        output_path: str | Path | None = None,
    ) -> Path:
        """Generate a PDF prescription from a pipeline *result* dict
        (as returned by ``run()``).

        Parameters
        ----------
        result : dict
            Output of ``self.run(text)``.
        patient_name : str
            Name to print on the prescription.
        output_path : str or Path, optional
            Where to save the PDF.  Defaults to
            ``output/prescription_<timestamp>.pdf``.

        Returns
        -------
        Path
            Absolute path to the generated PDF file.
        """
        if output_path is None:
            output_dir = _BASE_DIR / "output"
            output_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"prescription_{ts}.pdf"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # ---- Header ----
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 12, "First Doctor", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(
            0,
            6,
            "AI Triage & Preliminary Diagnostics (Student Project)",
            new_x="LMARGIN",
            new_y="NEXT",
            align="C",
        )
        pdf.ln(4)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

        # ---- Patient info ----
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"Patient: {patient_name}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(
            0,
            7,
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(4)

        # ---- Extracted symptoms ----
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Extracted Symptoms", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        symptoms = result.get("extracted_symptoms", [])
        if symptoms:
            for sym in symptoms:
                pdf.cell(
                    0,
                    6,
                    f"  - {sym.replace('_', ' ').title()}",
                    new_x="LMARGIN",
                    new_y="NEXT",
                )
        else:
            pdf.cell(0, 6, "  (none detected)", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # ---- Predictions ----
        for i, pred in enumerate(result.get("predictions", []), start=1):
            disease = pred["disease"]
            prob = pred["probability"] * 100

            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(
                0,
                8,
                f"Diagnosis #{i}: {disease}  ({prob:.1f}%)",
                new_x="LMARGIN",
                new_y="NEXT",
            )

            # Known symptoms for this disease
            known = pred.get("known_symptoms", [])
            if known:
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(
                    0,
                    6,
                    f"  Associated symptoms: {', '.join(s.replace('_', ' ') for s in known[:10])}"
                    + (" ..." if len(known) > 10 else ""),
                    new_x="LMARGIN",
                    new_y="NEXT",
                )

            # Recommended tests
            rec = pred.get("recommended_tests", {})
            tests = rec.get("tests", [])
            if tests:
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(
                    0, 6, "  Recommended Tests:", new_x="LMARGIN", new_y="NEXT"
                )
                pdf.set_font("Helvetica", "", 10)
                for test in tests:
                    pdf.cell(
                        0,
                        6,
                        f"    * {test}",
                        new_x="LMARGIN",
                        new_y="NEXT",
                    )

            # Escalation info
            esc = rec.get("escalation")
            if esc:
                pdf.set_font("Helvetica", "B", 10)
                color = (
                    (220, 20, 20)
                    if esc["priority"] == "EMERGENCY"
                    else (200, 120, 0)
                    if esc["priority"] in ("HIGH", "URGENT")
                    else (0, 0, 0)
                )
                pdf.set_text_color(*color)
                pdf.cell(
                    0,
                    6,
                    f"  !! {esc['priority']}: {esc['notes']}",
                    new_x="LMARGIN",
                    new_y="NEXT",
                )
                pdf.set_text_color(0, 0, 0)

            pdf.ln(3)

        # ---- Disclaimer ----
        pdf.ln(6)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(
            0,
            5,
            "DISCLAIMER: This is an AI-generated preliminary report from a "
            "student project.  It is NOT a medical diagnosis.  Please consult "
            "a certified medical professional before taking any action.",
        )

        pdf.output(str(output_path))
        return output_path.resolve()


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    detector = DiseaseDetector()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Describe your symptoms: ")

    print("\n[Phase 1] Extracting symptoms …")
    result = detector.run(text)

    print(f"  Detected: {result['extracted_symptoms']}")

    print("\n[Phase 2 & 3] Predictions:")
    for i, p in enumerate(result["predictions"], 1):
        print(f"  #{i}  {p['disease']}  —  {p['probability']*100:.1f}%")
        tests = p["recommended_tests"].get("tests", [])
        if tests:
            print(f"       Tests: {', '.join(tests)}")
        esc = p["recommended_tests"].get("escalation")
        if esc:
            print(f"       ⚠ {esc['priority']}: {esc['notes']}")

    print("\n[Phase 4] Generating PDF …")
    pdf_path = detector.generate_pdf(result, patient_name="Test Patient")
    print(f"  Saved to: {pdf_path}")
