---
title: Implementation of a Discriminative Model (NFL Position Classification)
parent: AI, High Performance Computing, and Ethical Considerations
nav_order: 3
---

### Overview
Developed a **hybrid classification system** combining a Random Forest classifier with a PyTorch neural network to predict NFL player positions from historical performance data.

### Methodologies
- Feature encoding and scaling
- Random Forest with GridSearchCV
- Neural network incorporating Random Forest predictions
- Confusion matrix analysis

### Tools

| Tool | Use |
|---|---|
| Scikit‑learn | Random Forest + GridSearchCV |
| PyTorch | Neural network |
| Pandas / NumPy | Data manipulation |
| Seaborn | Confusion matrix visualization |

### Data Cleaning and EDA
```python
data_cleaned = df.dropna()
data_cleaned['Longest Reception'] = data_cleaned['Longest Reception'].str.replace('T','')
data_cleaned['Longest Reception'] = data_cleaned['Longest Reception'].str.replace('F','')
data_cleaned['Longest Reception'] = pd.to_numeric(data_cleaned['Longest Reception'], errors='coerce', downcast='integer')
data_cleaned=data_cleaned.drop(['Player Id', 'Name', 'Team', 'Year','Games Played','Fumbles'], axis=1)
data_cleaned=data_cleaned.dropna()
data_cleaned['Position'].value_counts()
dropping_outliers = ['CB','DE','FS','DB','SS','LB']
data_cleaned = data_cleaned[~data_cleaned['Position'].isin(dropping_outliers)]
data_cleaned['Position'].value_counts()
LabelEncoder = LabelEncoder()
data_cleaned['Position'] = LabelEncoder.fit_transform(data_cleaned['Position'])
data_cleaned['Receptions'] = LabelEncoder.fit_transform(data_cleaned['Receptions'])
data_cleaned['Receiving Yards'] = LabelEncoder.fit_transform(data_cleaned['Receiving Yards'])
data_cleaned['Yards Per Reception'] = LabelEncoder.fit_transform(data_cleaned['Yards Per Reception'])
data_cleaned['Receiving TDs'] = LabelEncoder.fit_transform(data_cleaned['Receiving TDs'])
data_cleaned['Receptions Longer than 20 Yards'] = LabelEncoder.fit_transform(data_cleaned['Receptions Longer than 20 Yards'])
data_cleaned['Receptions Longer than 40 Yards'] = LabelEncoder.fit_transform(data_cleaned['Receptions Longer than 40 Yards'])
data_cleaned['First Down Receptions'] = LabelEncoder.fit_transform(data_cleaned['First Down Receptions'])
correlation_map = sns.heatmap(data_cleaned.corr(numeric_only=True), cmap="coolwarm", annot=True)
select = SelectKBest(score_func=f_regression, k='all')
x_select = data_cleaned.drop("Position", axis=1)
y_select = data_cleaned['Position']
select.fit(x_select,y_select)
feature_scores = pd.DataFrame({'Feature': x_select.columns, 'Score': select.scores_})
print(feature_scores.sort_values(by='Score', ascending=False))
y = data_cleaned['Position']
data_cleaned = data_cleaned.drop(['Receiving Yards','First Down Receptions','Receptions','Position'], axis=1)
normalize = StandardScaler()
data_cleaned = normalize.fit_transform(data_cleaned)
x = data_cleaned
y = y.values.reshape(-1,1)
print(data_cleaned)
print(data_cleaned.shape)
```
### Pytorch Classes and Training
```python
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
print(x_train.shape, x_val.shape, x_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class NFLDataset(Dataset):
    def __init__(self, x, y, forest_pred):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.long).to(device)
        self.forest_pred = torch.tensor(forest_pred, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.forest_pred[idx]
forest_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

forest = RandomForestClassifier()
grid_search = GridSearchCV(forest, forest_param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train.ravel())
forest = grid_search.best_estimator_
print("Best Random Forest Parameters: ", grid_search.best_params_)
print("Best Random Forest Accuracy: ", grid_search.best_score_)
forest_pred_train = forest.predict(x_train)
forest_pred_val = forest.predict(x_val)
forest_pred_test = forest.predict(x_test)
class NFLModelWithForest(nn.Module):
    def __init__(self, input_size, forest_output_size, hidden_size1, hidden_size2):
        super(NFLModelWithForest, self).__init__()
        self.fc1 = nn.Linear(input_size + forest_output_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 5)
        self.relu = nn.ReLU()

    def forward(self, x, forest_pred):
        x = torch.cat((x, forest_pred.unsqueeze(1)), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
hyper_grid_loop = {
    'hidden_size1': [16, 32, 64],
    'hidden_size2': [16, 32, 64],
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'epochs': [50, 100, 200]
}
train_dataset = NFLDataset(x_train, y_train, forest_pred_train)
valid_dataset = NFLDataset(x_val, y_val, forest_pred_val)
test_dataset = NFLDataset(x_test, y_test, forest_pred_test)

best_model = None
best_loss = float('inf')
best_hyperparameters = None
input_size = x_train.shape[1]
for hidden_size1, hidden_size2, learning_rate, batch_size, epochs in product(*hyper_grid_loop.values()):
    forest_output_size = 1
    model = NFLModelWithForest(input_size,forest_output_size, hidden_size1,hidden_size2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for x_batch, y_batch, forest_pred_batch in train_loader:
            optimizer.zero_grad()
            y_batch = y_batch.view(-1)
            y_pred = model(x_batch, forest_pred_batch)
            loss = criterion(y_pred, y_batch.long()) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)    

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for x_batch, y_batch, forest_pred_batch in valid_loader:
                y_batch = y_batch.view(-1)
                y_pred = model(x_batch, forest_pred_batch)
                valid_loss += criterion(y_pred, y_batch)
            avg_valid_loss = valid_loss / len(valid_loader)
        print(f"Epoch: {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}")
        print(f"Current Parameters: hidden_size1={hidden_size1}, hidden_size2={hidden_size2}, learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")
        
        if valid_loss < best_loss:
            best_model = model
            best_loss = valid_loss
            best_hyperparameters = (hidden_size1, hidden_size2, learning_rate, batch_size, epochs)
            
print(f"Best Validation Loss: {best_loss:.4f}")
print(f"Best Hyperparameters: hidden_size1={best_hyperparameters[0]}, hidden_size2={best_hyperparameters[1]}, learning_rate={best_hyperparameters[2]}, batch_size={best_hyperparameters[3]}, epochs={best_hyperparameters[4]}")
```
### Model Eval
```python
model.eval()
predictions = []
true_class = []
with torch.no_grad():
    for x_batch, y_batch, forest_pred_batch in test_loader:
        y_batch = y_batch.view(-1)  
        y_pred = model(x_batch, forest_pred_batch)
        predictions.extend(torch.argmax(y_pred, dim=1).cpu().numpy())
        true_class.extend(y_batch.cpu().numpy())

original_class_labels = ['QB', 'FB', 'RB', 'TE', 'WR']
class_idx_to_label = {idx: label for idx, label in enumerate(original_class_labels)}

predictions_labels = [class_idx_to_label[pred] for pred in predictions]
true_class_labels = [class_idx_to_label[label] for label in true_class]
# Confusion Matrix
cm = confusion_matrix(true_class_labels, predictions_labels, labels=original_class_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=original_class_labels)
disp.plot()
plt.show()

correct_predictions = sum(p == t for p, t in zip(predictions_labels, true_class_labels))
accuracy = correct_predictions / len(true_class_labels)
print(f"Accuracy: {accuracy:.4f}")
```
### Outcomes
- **Accuracy:** 62.3%
- Strong performance for wide receivers
- Identified class imbalance as a key limitation
- [Project Link](https://github.com/jeffmoe/jeffmoe.github.io/blob/main/Project%20Docs/Discriminative_Model.ipynb)
<p><img width="545" height="481" alt="image" src="https://github.com/user-attachments/assets/c29c2b1d-82c1-4528-944f-e8aeac4c9317" /></p>
---
