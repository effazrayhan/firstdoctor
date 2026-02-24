# First Doctor

## How to read the model

1. Read model from the `.pth` file

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SymptomClassifier(X.shape[1], len(le.classes_))
model.load_state_dict(torch.load("torch_symptom_model.pth", map_location=device))
model.to(device)
model.eval()
```

Here, you might need this setup for problem and symptomps header
```python
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Convert DataFrame to NumPy first
X = df.drop(columns=['diseases']).to_numpy()
y = le.fit_transform(df['diseases'])
```
---
2. Write a predict diseases function
```python
import numpy as np

def predict_top_diseases(symptoms_vector, top_k=3):
    listOfSymptomsHeader = df.columns[:-1].tolist()  # all columns except 'diseases'
    for i in range(len(symptoms_vector)):
        if symptoms_vector[i] == 1:
            print(f"Symptom: {listOfSymptomsHeader[i]} is present.")
    print("\nPredicting top diseases based on symptoms...")
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(symptoms_vector, dtype=torch.float32).to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=0).cuda(device=device).cpu().numpy()  # move to CPU and convert to NumPy

    # Sort probabilities in descending order
    sorted_indices = np.argsort(probs)[::-1]
    top_indices = sorted_indices[:top_k]

    # Build list of (disease, probability)
    top_diseases = [(le.classes_[i], probs[i]) for i in top_indices]
    return top_diseases
```
---
3. Now this is how you will get final topK(default=3) predicted disease/problem according to the symptomps
```python
sample = [] # object input of array of symptops
top_results = predict_top_diseases(sample, top_k=4)

print(f"\nTop probable diseases for sample {i}:")
    
for disease, prob in top_results:
    print(f"{disease}: {prob:.2f}")
```
