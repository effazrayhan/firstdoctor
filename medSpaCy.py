import medspacy
nlp = medspacy.load()
doc = nlp("Patient reports acute chest pain and shortness of breath.")

for ent in doc.ents:
    print(f"Symptom: {ent.text}, Label: {ent.label_}")