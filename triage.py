EMERGENCY_KEYWORDS = [
    "chest pain", "difficulty breathing", "unconscious", 
    "severe bleeding", "stroke symptoms", "suicide", "poison","heart attack", "cardiac arrest", "no pulse", "irregular heartbeat",
"heart is racing", "collapsed suddenly", "tightness in chest",

"can’t breathe", "shortness of breath", "gasping for air",
"choking", "airway blocked", "turning blue", "asthma attack",

"seizure", "convulsions", "fainted", "not responding",
"sudden confusion", "paralysis", "numbness on one side",
"blurred vision", "sudden severe headache",

"heavy bleeding", "bleeding won’t stop", "deep cut",
"gunshot wound", "stab wound", "broken bone",
"head injury", "car accident", "road accident",
"burn injury", "electrocution",

"overdose", "took too many pills", "swallowed poison",
"chemical exposure", "gas leak", "carbon monoxide poisoning",

"anaphylaxis", "severe allergic reaction",
"face swelling", "throat swelling",
"high fever 104", "heat stroke", "hypothermia",

"panic attack severe", "want to die", "self harm",
"ending it", "no reason to live"
]

def check_emergency(user_input):
    input_lower = user_input.lower()
    for word in EMERGENCY_KEYWORDS:
        if word in input_lower:
            return True
    return False