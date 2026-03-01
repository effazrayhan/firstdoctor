# First Doctor — Project Brief

## Overview
**First Doctor** is a Disease Detection Pipeline — a student-built AI triage and preliminary diagnostics system. It takes natural language symptom descriptions from users, extracts medical entities, predicts diseases using a pre-trained model, and maps predictions to recommended laboratory tests.

## Core Pipeline Flow
1. **User Input** → Natural language symptom description
2. **Emergency Triage** → Keyword-based emergency detection (`triage.py`)
3. **Clinical Interview** → LLM-driven follow-up questions (Ollama/Groq via `brain.py`, `app.py`, `start_consultation.py`)
4. **Symptom Extraction** → spaCy/medspaCy NLP entity recognition (`medspacy.py`, `medSpaCy.py`, `predictor.py`)
5. **Disease Prediction** → PyTorch SymptomClassifier model (`model/modelrun.py`, `predictor.py`)
6. **Test Mapping** → Fuzzy matching + keyword matching to test bundles (`fuzzy.py`, `start_consultation.py`, `database.py`)
7. **Escalation** → Rule-based escalation logic (`main.py`)

## Tech Stack
- **Frontend:** Streamlit (`app.py`)
- **LLM Backend:** Ollama (llama3.2:3b) / Groq Cloud API
- **NLP:** medspaCy for medical entity extraction
- **ML Model:** PyTorch SymptomClassifier (saved as `torch_symptom_model.pth`)
- **Matching:** thefuzz for fuzzy disease-to-test mapping
- **Data:** SQLite (`database.py`), JSON test bundles (`test_bundles.json`)
- **Training utilities:** sklearn LabelEncoder, pandas, numpy
