import json

def load_escalation_rules(path="test_bundles.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data["escalation_logic"]

def check_trigger(trigger, symptoms, labs):
    import re
    trigger_lower = trigger.lower()

    # Create a case-insensitive labs lookup dict
    labs_lower = {k.lower(): v for k, v in labs.items()}

    # Simple numeric rule parsing
    if ">" in trigger_lower:
        field, threshold_str = trigger_lower.split(">", 1)
        field = field.strip()
        # Extract just the numeric part (handles "5 days", "9%", etc.)
        numbers = re.findall(r'[\d.]+', threshold_str)
        if numbers:
            threshold = float(numbers[0])
            return labs_lower.get(field, 0) > threshold
        # If no number found, treat as text match
        return any(trigger_lower in s.lower() for s in symptoms)

    if "<" in trigger_lower:
        field, threshold_str = trigger_lower.split("<", 1)
        field = field.strip()
        # Extract just the numeric part
        numbers = re.findall(r'[\d.]+', threshold_str)
        if numbers:
            threshold = float(numbers[0])
            return labs_lower.get(field, 9999) < threshold
        # If no number found, treat as text match
        return any(trigger_lower in s.lower() for s in symptoms)

    # Text match for simple descriptions
    return any(trigger_lower in s.lower() for s in symptoms)

def evaluate_escalation(symptoms, labs):
    rules = load_escalation_rules()
    activated_rules = []

    for rule_name, rule_data in rules.items():
        triggers = rule_data["trigger_conditions"]

        for trigger in triggers:
            if check_trigger(trigger, symptoms, labs):
                activated_rules.append({
                    "rule": rule_name,
                    "priority": rule_data["action"]["priority"],
                    "recommend_tests": rule_data["action"]["recommend_tests"],
                    "notes": rule_data["action"]["notes"]
                })
                break  # stop after first trigger match

    return activated_rules

def main():
    print("=== Escalation Logic Tester ===")

    symptoms_input = input("Enter symptoms (comma separated): ")
    symptoms = [s.strip() for s in symptoms_input.split(",")]

    # Example lab input (manual for now)
    labs = {
        "Hemoglobin": 9,
        "HbA1c": 10,
        "Serum Creatinine": 1.2
    }

    results = evaluate_escalation(symptoms, labs)

    if not results:
        print("\nNo escalation rules triggered.")
    else:
        print("\nActivated Escalations:\n")
        for r in results:
            print(f"Rule: {r['rule']}")
            print(f"Priority: {r['priority']}")
            print(f"Recommended Tests: {', '.join(r['recommend_tests'])}")
            print(f"Notes: {r['notes']}")
            print("-" * 40)

if __name__ == "__main__":
    main()