def diagnose_user(text):
    # 1. Extract symptoms from natural language
    doc = nlp(text)
    detected = [ent.text.lower() for ent in doc.ents]
    
    # 2. Create the binary vector (0s and 1s)
    input_vector = np.zeros(len(symptom_names))
    for i, sym in enumerate(symptom_names):
        clean_sym = sym.replace('_', ' ').lower()
        if clean_sym in detected:
            input_vector[i] = 1
            
    # 3. Predict Disease
    prediction = model.predict(np.array([input_vector]))
    disease_idx = np.argmax(prediction)
    disease_name = encoder.classes_[disease_idx]
    
    # 4. Map to Test Bundles (Goal 2)
    print(f"Prediction: {disease_name}")
    # Call your get_diagnostic_plan(disease_name) here!