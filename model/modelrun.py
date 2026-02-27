# %%

import pandas as pd # data processing, CSV file
import numpy as np
import os
pName = ""
for dirname, _, filenames in os.walk('./database'):
    for filename in filenames:
        pName = (os.path.join(dirname, filename))
        print(pName)
        
if pName != "":
    print('Data source import complete.')
else:
    print('Data source import failed.')


# %% [markdown]
# ## Now I will split dataset into X and y and train a model using sklearn LabelEncoder
# 

# %%
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(pName)

le = LabelEncoder()

# Convert DataFrame to NumPy first
X = df.drop(columns=['diseases']).to_numpy()
y = le.fit_transform(df['diseases'])

# Now convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Dataset + DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
# Define the same architecture you used before
class SymptomClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SymptomClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # logits

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SymptomClassifier(X.shape[1], len(le.classes_))
model.load_state_dict(torch.load("torch_symptom_model.pth", map_location=device))
model.to(device)
model.eval()

# %%
def predict_top_diseases(symptoms_vector, top_k=3):
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

# %%
allResults = []
randomizedData = np.random.permutation(X)  # Shuffle the dataset
for i in range(0,15):
    sample = X[i] 
    
    top_results = predict_top_diseases(sample)
    allResults.append((i+1, top_results))
        

# %% [markdown]
# ## YOU HAVE TO RECEIVE DATA FROM `FRONTEND` INSTEAD OF THIS and STORE INTO THE VARIABLE `{sample}` , Then don't need to run the loop, just call the function `predict_top_diseases(symptoms_vector, top_k=3)` once with the received data.
# 

# %%
for i, res in allResults:
    row = ""
    for disease, prob in res:
        row += f"\t{disease}: {(prob*100):.2f}%\n"
    print("-"*(row.__len__()+15))
    print(f"{i} : {row}")


