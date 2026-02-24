import sqlite3

def init_db():
    conn = sqlite3.connect('first_doctor.db')
    cursor = conn.cursor()
    
    # 1. Patient Table
    cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY, name TEXT, age INTEGER)''')
    
    # 2. Consultation Table (Tracks the AI session)
    cursor.execute('''CREATE TABLE IF NOT EXISTS consultations (
        id INTEGER PRIMARY KEY, 
        patient_id INTEGER, 
        raw_symptoms TEXT, 
        detected_category TEXT, 
        test_list TEXT,
        status TEXT DEFAULT 'Pending Results',
        FOREIGN KEY(patient_id) REFERENCES patients(id))''')
    
    conn.commit()
    conn.close()

def get_tests_for_symptom(user_input):
    """
    Logic to match keywords from user input to our expanded JSON.
    """
    import json
    with open('test_bundles.json', 'r') as f:
        data = json.load(f)
    
    match_list = []
    category = "General"
    
    input_text = user_input.lower()
    for cat, details in data.items():
        if any(key in input_text for key in details['keywords']):
            match_list.extend(details['tests'])
            category = cat
            
    return list(set(match_list)), category