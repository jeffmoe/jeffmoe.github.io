### Overview
Built a regression model to predict **food delivery times** using factors such as distance, preparation time, and weather conditions. Model performance was assessed using **R², RMSE, and MAE**.

### Methodologies
- Data cleaning and encoding
- Feature selection with correlation analysis and `SelectKBest`
- Standardization using `StandardScaler`
- PyTorch regression network
- Residual analysis for model validation

### Tools

| Tool | Use |
|---|---|
| PyTorch | Neural network model |
| Pandas | Data cleaning |
| Scikit‑learn | Feature selection & metrics |
| Seaborn | Correlation heatmaps |
| Matplotlib | Residual analysis |
| CUDA | GPU acceleration |

### Data Cleaning and EDA
```python
data_clean = data.drop(['Order_ID'], axis=1)
labels = LabelEncoder()
data_clean['Weather'] = labels.fit_transform(data_clean['Weather'])
data_clean['Traffic_Level'] = labels.fit_transform(data_clean['Traffic_Level'])
data_clean['Time_of_Day'] = labels.fit_transform(data_clean['Time_of_Day'])
data_clean['Vehicle_Type'] = labels.fit_transform(data_clean['Vehicle_Type'])
data_clean = data_clean.dropna(axis=0,how='any')
correlation_map = sns.heatmap(data_clean.corr(numeric_only=True), cmap="coolwarm", annot=True)
select = SelectKBest(score_func=f_regression, k='all')
x_select = data_clean.drop("Delivery_Time_min", axis=1)
y_select = data_clean['Delivery_Time_min']
select.fit(x_select,y_select)
feature_scores = pd.DataFrame({'Feature': x_select.columns, 'Score': select.scores_})
print(feature_scores.sort_values(by='Score', ascending=False))
data_clean = data_clean.drop(['Courier_Experience_yrs','Traffic_Level','Time_of_Day','Vehicle_Type'],axis=1)
target = data_clean['Delivery_Time_min']
data_clean = data_clean.drop('Delivery_Time_min', axis=1)
normalize = StandardScaler()
data_clean = normalize.fit_transform(data_clean)
print(data_clean)
print(data_clean.shape)
x = data_clean
y = target.values.reshape(-1,1)
```
### Pytoch Classes - Training and Testing
```python
class Time_Predict_Data(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
class Time_Predict_Reg(nn.Module):
    def __init__(self, input_size, layer1, layer2, layer3):
        super(Time_Predict_Reg, self).__init__()
        self.linear = nn.Linear(input_size, layer1)
        self.linear1 = nn.Linear(layer1,layer2)
        self.linear2 = nn.Linear(layer2,layer3)
        self.linear3 = nn.Linear(layer3,1)

    def forward(self, x):
        x = torch.relu(self.linear(x))
        x = torch.relu(self.linear1(x)) 
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return x
train_dataset = Time_Predict_Data(x_train, y_train)
valid_dataset = Time_Predict_Data(x_val, y_val)
test_dataset = Time_Predict_Data(x_test, y_test)
best_model = None
best_loss = float('inf')
best_params = None
input_size = x_train.shape[1]

parameters = {
    'layer1': [32, 64, 128],
    'layer2': [32, 64, 128],
    'layer3': [32, 64, 128],
    'lr': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'epochs': [100, 200, 300]
}
for batch_size, layer1, layer2, layer3, lr, epochs in product(
        parameters['batch_size'],
        parameters['layer1'],
        parameters['layer2'],
        parameters['layer3'],
        parameters['lr'],
        parameters['epochs']):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    model = Time_Predict_Reg(input_size, layer1, layer2, layer3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for x_val, y_val in valid_loader:
                y_pred = model(x_val)
                loss = criterion(y_pred, y_val)
                valid_loss += loss.item()
            valid_loss /= len(valid_loader)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = model
            best_params = (layer1, layer2, layer3, lr, batch_size, epochs)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, current best loss: {best_loss}, best params: {best_params}")
print(f"Train Loss: {train_loss}, Valid Loss: {valid_loss}, layer1: {best_params[0]}, layer2: {best_params[1]}, layer3: {best_params[2]}, lr: {best_params[3]}, batch_size: {best_params[4]}, epochs: {best_params[5]}")
test_loader = DataLoader(test_dataset, batch_size=best_params[4], shuffle=False)
best_model.eval()
Time_Guess = []
Time_Actual = []

with torch.no_grad():
    for x_test, y_test in test_loader:
        y_pred = best_model(x_test)
        Time_Guess.extend(y_pred.numpy())
        Time_Actual.extend(y_test.numpy())

Time_Guess = np.array(Time_Guess).reshape(-1,1)
Time_Actual = np.array(Time_Actual).reshape(-1,1)
R2 = r2_score(Time_Actual, Time_Guess)
coefficients = best_model.linear.weight.detach().numpy()
intercept = best_model.linear.bias.detach().numpy()

linear_df = pd.DataFrame(coefficients, columns= ['feature 1', 'feature 2', 'feature 3'])
linear_df['Intercept'] = intercept

linear_df
avg_feature1 = linear_df['feature 1'].mean()
avg_feature2 = linear_df['feature 2'].mean()
avg_feature3 = linear_df['feature 3'].mean()
avg_intercept = linear_df['Intercept'].mean()
print(avg_feature1, avg_feature2, avg_feature3, avg_intercept)
```
### Outcomes
- **R²:** 0.812
- **RMSE:** 9.85 minutes
- **MAE:** 7.30 minutes
- Identified prediction weaknesses with longer delivery times
  [Project Link]
<p><img width="666" height="568" alt="image" src="https://github.com/user-attachments/assets/4fccc560-75da-4acb-8ff7-c018c98bdf7c" /></p>
<p><img width="852" height="545" alt="image" src="https://github.com/user-attachments/assets/4d37a7fa-39fc-47ec-8b1f-93f099772133" /></p>
<p><img width="843" height="548" alt="image" src="https://github.com/user-attachments/assets/5bd0b61c-008d-4729-8a55-a40d21f2c325" /></p>
<p><img width="788" height="597" alt="image" src="https://github.com/user-attachments/assets/64bbcb96-be07-41ce-b713-f98540a617bf" /></p>
---
