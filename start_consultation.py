import json
import os
from triage import check_emergency
from brain import get_medical_advice, SYSTEM_PROMPT
import ollama

# Load the test bundles we created in 1.4
with open('test_bundles.json', 'r') as f:
    TEST_DATA = json.load(f)

def get_tests_from_text(text):
    """Scans the conversation for keywords to pull tests from JSON."""
    detected_tests = []
    category = "General Health"
    text = text.lower()
    
    for cat, details in TEST_DATA.items():
        if any(key in text for key in details['keywords']):
            detected_tests.extend(details['tests'])
            category = cat
    return list(set(detected_tests)), category

def run_first_doctor():
    print("--- FIRST DOCTOR: AI TRIAGE SYSTEM (STUDENT PROJECT) ---")
    print("Type 'quit' to exit at any time.\n")
    
    # 1. Start Conversation
    user_input = input("First Doctor: Hello. Please describe what you are feeling today.\nYou: ")
    
    if user_input.lower() == 'quit': return

    # 2. Immediate Emergency Check
    if check_emergency(user_input):
        print("\n🚨 FIRST DOCTOR ALERT: Your symptoms suggest an emergency. Please call 911 or go to the nearest ER immediately.")
        return

    # 3. The Interview Loop
    chat_history = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_input}
    ]
    
    # We will ask 3 follow-up questions total
    for i in range(3):
        print("\n[Thinking...]")
        response = ollama.chat(model='llama3.2:3b', messages=chat_history)
        ai_message = response['message']['content']
        
        # Add AI response to history
        chat_history.append({'role': 'assistant', 'content': ai_message})
        
        print(f"First Doctor: {ai_message}")
        
        user_answer = input("You: ")
        if user_answer.lower() == 'quit': return
        
        # Add User answer to history
        chat_history.append({'role': 'user', 'content': user_answer})

    # 4. Final Processing & Test Generation
    print("\n" + "="*50)
    print("FIRST DOCTOR: PRELIMINARY TEST RECOMMENDATION")
    print("="*50)

    # Combine all user answers to find keywords
    full_conversation = " ".join([m['content'] for m in chat_history if m['role'] == 'user'])
    suggested_tests, category = get_tests_from_text(full_conversation)

    print(f"Detected Category: {category.replace('_', ' ').title()}")
    print("\nBased on our conversation, you should consider these tests:")
    
    if not suggested_tests:
        print("- Routine Physical Examination")
        print("- General Blood Profile (CBC)")
    else:
        for test in suggested_tests:
            print(f"✅ {test}")

    print("\n" + "-"*50)
    print("IMPORTANT: This is an AI-generated suggestion for a student project.")
    print("Please present this list to a certified medical professional.")
    print("-"*50)

if __name__ == "__main__":
    run_first_doctor()