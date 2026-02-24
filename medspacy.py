import medspacy
from medspacy.target_matcher import TargetRule

# Get symptom names from columns
symptom_names = df.columns[1:].tolist()

# Create rules automatically
nlp = medspacy.load()
matcher = nlp.get_pipe("medspacy_target_matcher")

rules = [TargetRule(sym.replace('_', ' '), "SYMPTOM") for sym in symptom_names]
matcher.add(rules)