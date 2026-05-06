---
---
<link rel="stylesheet" href="styles.css">

# Data Science, Cloud, and Machine Learning Portfolio

This portfolio highlights graduate‑level projects spanning **data science, machine learning, cloud architecture, databases, enterprise architecture, and risk management**. The work demonstrates hands‑on experience with industry‑standard tools, model development, infrastructure design, and real‑world problem solving.

---

## Table of Contents

1. [AI, High Performance Computing, and Ethical Considerations](#ai-high-performance-computing-and-ethical-considerations)
   - [Develop and Analyze a GAN and Classifier](#project-develop-and-analyze-a-gan-and-classifier)
   - [Develop and Analyze a Linear Regression Model](#project-develop-and-analyze-a-linear-regression-model)
   - [Implementation of a Discriminative Model (NFL Position Classification)](#project-implementation-of-a-discriminative-model-nfl-position-classification)

2. [Cloud Architecture and Infrastructure](#cloud-architecture-and-infrastructure)
   - [AWS IaC Deployment with Serverless Alerting](#project-aws-iac-deployment-with-serverless-alerting)
   - [Deploy a Secure Web Application on AWS](#project-deploy-a-secure-web-application-on-aws)

3. [Database Systems](#database-systems)
   - [AI‑Enhanced Database Ecosystem (AWS)](#project-ai-enhanced-database-ecosystem-aws)
   - [Time Series Database for DevOps Monitoring](#project-time-series-database-for-devops-monitoring)

4. [Enterprise Architecture, Strategy, and Risk](#enterprise-architecture-strategy-and-risk)
   - [Unified Multi‑Domain Enterprise Architecture (TOGAF + Zero Trust)](#project-unified-multi-domain-enterprise-architecture-togaf--zero-trust)

5. [Project Management, Systems Development, and Risk](#project-management-systems-development-and-risk)
   - [AI Risk Mitigation Plan for E‑commerce Platform](#project-ai-risk-mitigation-plan-for-e-commerce-platform)

6. [Machine Learning and Artificial Intelligence](#machine-learning-and-artificial-intelligence)
   - [LSTM Stock Price Prediction (TensorFlow)](#project-lstm-stock-price-prediction-tensorflow)
   - [Real‑Time Machine Learning Pipeline](#project-real-time-machine-learning-pipeline)

7. [Summary](#summary)

---

## AI, High Performance Computing, and Ethical Considerations

Focused on deep learning, parallel computing, and ethical considerations in AI, with hands‑on projects using PyTorch and modern ML workflows.

---

## Project: Develop and Analyze a GAN and Classifier

### Overview
Developed a **CNN classifier** and **GAN** using the MNIST dataset (70,000 handwritten digits). The classifier achieved high accuracy, while the GAN successfully generated lifelike synthetic digits. Training was optimized using **multi‑GPU parallelization** and **Automatic Mixed Precision (AMP)**.

### Project Link

### Methodologies
- CNN‑based digit classifier
- GAN architecture with generator and discriminator
- PyTorch `DataParallel`
- Mixed precision (AMP)
- Performance metrics and visual analysis

### Tools

| Tool         | Use                            |
|:-------------|:-------------------------------|
| PyTorch      | Model development and training |
| Torchvision  | Dataset loading                |
| Matplotlib   | Visualization                  |
| NumPy        | Data processing                |
| Scikit‑learn | Evaluation metrics             |
| psutil / os  | Resource monitoring            |

### Custom Pytorch Classes
```python
class MNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        self.mnist_data = datasets.MNIST(root=root, train=train, transform=transform, download=download)

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        image, label = self.mnist_data[idx]
        return image, label
    

class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super(Generator, self).__init__()
        self.img_dim = img_dim
        self.fc = nn.Linear(noise_dim, 128 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.deconv(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1) 
        )

    def forward(self, x):
        return self.model(x)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 14 * 14) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
### Training Loop
```python
num_epochs = 70
batch_size = 128
learning_rate = 0.0002
noise_dim = 100
img_dim = (1, 28, 28)
G_losses = []
D_losses = []
C_losses = 0.0
img_list = []
precision_list = []
recall_list = []
fixed_noise = torch.randn(32, noise_dim).to(device)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=24)


classifer = Classifier().to(device)
generator = Generator(noise_dim,img_dim).to(device)
discriminator = Discriminator().to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)
    classifer = nn.DataParallel(classifer)

criterion = nn.BCEWithLogitsLoss()
criterion_classifer = nn.CrossEntropyLoss()
optimizer_c = optim.SGD(classifer.parameters(), lr = learning_rate, momentum=.9)
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

scaler = GradScaler()

for epoch in range(num_epochs):
    for i, (real_images, labels) in enumerate(train_loader):
        real_images = real_images.to(device)
        labels = labels.to(device)
        current_batch_size = real_images.size(0)
        real_labels = torch.ones(current_batch_size, 1, device=device, dtype=torch.float)
        fake_labels = torch.zeros(current_batch_size, 1, device=device, dtype=torch.float)

        optimizer_d.zero_grad()
        with autocast(device_type='cuda'):
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)
            noise = torch.randn(current_batch_size, noise_dim).to(device)
            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss = d_loss_real + d_loss_fake
        scaler.scale(d_loss).backward()
        scaler.step(optimizer_d)
        
        optimizer_g.zero_grad()
        with autocast(device_type='cuda'):
            noise = torch.randn(current_batch_size, noise_dim).to(device)
            fake_images = generator(noise)
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
        scaler.scale(g_loss).backward()
        scaler.step(optimizer_g)

        D_losses.append(d_loss.item())
        G_losses.append(g_loss.item())

        optimizer_c.zero_grad()
        with autocast(device_type='cuda'):
            outputs = classifer(real_images)
            c_loss = criterion_classifer(outputs, labels)
        scaler.scale(c_loss).backward()
        scaler.step(optimizer_c)

        scaler.update()

        C_losses +=c_loss.item()

        _,predicted = torch.max(outputs,1)
        precision = precision_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
        recall = recall_score(labels.cpu(), predicted.cpu(), average='macro',zero_division=0)
        precision_list.append(precision)
        recall_list.append(recall)

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], 'f'D Loss: {d_loss.item():.3f}, G Loss: {g_loss.item():.3f}, C Loss: {C_losses / 100:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}')
            C_losses = 0.0
        if (i % 500) == 0:
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    c_loss = 0.0
```
### Outcomes
- **Classifier accuracy:** 97%
- High precision and recall across all classes
- GAN converged with decreasing generator and discriminator losses
- Substantial reduction in training time using AMP and parallelism

<p><img width="846" height="468" alt="image" src="https://github.com/user-attachments/assets/a5acc71a-0db1-4eca-8b15-7db6886c742d" /></p>
<p><img width="833" height="468" alt="image" src="https://github.com/user-attachments/assets/ff4cb7c2-ed8e-487c-94c1-757e64a7dae2" /></p>
<p><img width="1182" height="568" alt="image" src="https://github.com/user-attachments/assets/050ca4d9-431a-4401-bb69-e4ad4a9db870" /></p>
---

## Project: Develop and Analyze a Linear Regression Model

### Overview
Built a regression model to predict **food delivery times** using factors such as distance, preparation time, and weather conditions. Model performance was assessed using **R², RMSE, and MAE**.
### Project Link
[Link]
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
<p><img width="666" height="568" alt="image" src="https://github.com/user-attachments/assets/4fccc560-75da-4acb-8ff7-c018c98bdf7c" /></p>
<p><img width="852" height="545" alt="image" src="https://github.com/user-attachments/assets/4d37a7fa-39fc-47ec-8b1f-93f099772133" /></p>
<p><img width="843" height="548" alt="image" src="https://github.com/user-attachments/assets/5bd0b61c-008d-4729-8a55-a40d21f2c325" /></p>
<p><img width="788" height="597" alt="image" src="https://github.com/user-attachments/assets/64bbcb96-be07-41ce-b713-f98540a617bf" /></p>
---

## Project: Implementation of a Discriminative Model (NFL Position Classification)

### Overview
Developed a **hybrid classification system** combining a Random Forest classifier with a PyTorch neural network to predict NFL player positions from historical performance data.
### Project Link
[Link]

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
<p><img width="545" height="481" alt="image" src="https://github.com/user-attachments/assets/c29c2b1d-82c1-4528-944f-e8aeac4c9317" /></p>
---

## Cloud Architecture and Infrastructure

Hands‑on experience designing, deploying, and monitoring cloud infrastructure using AWS best practices.

---

## Project: AWS IaC Deployment with Serverless Alerting

### Overview
Automated the deployment of AWS infrastructure using **CloudFormation** and implemented a **serverless monitoring pipeline** to log EC2 termination events and trigger email alerts.

### Tools

| Tool | Use |
|---|---|
| AWS CloudFormation | Infrastructure as Code |
| AWS Lambda | Serverless processing |
| EventBridge | Event‑driven triggers |
| CloudWatch | Logs and alarms |
| Auto Scaling | EC2 scaling management |

### YAML
```yaml
 ASGLaunchTemplate:
    Type: AWS::EC2::LaunchTemplate
    Properties:
      LaunchTemplateName: ASGLaunchTemplate
      LaunchTemplateData:
        ImageId: ami-07a6f770277670015
        InstanceType: t2.micro
        NetworkInterfaces:
            - DeviceIndex: 0
              Groups:
              - !GetAtt ASGSecurityGroup.GroupId
Parameters:
    SubnetIDs:
        Type: List<AWS::EC2::Subnet::Id>
        Description: List of subnet IDs
Resources:
  ASGSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Allow SSH and HTTP access
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
 ASG:
    Type: AWS::AutoScaling::AutoScalingGroup
    Properties:
      AutoScalingGroupName: MyASG
      MinSize: 1
      DesiredCapacity: 1
      MaxSize: 2
      VPCZoneIdentifier: !Ref SubnetIDs
      LaunchTemplate:
        LaunchTemplateId: !Ref ASGLaunchTemplate
        Version: !GetAtt ASGLaunchTemplate.LatestVersionNumber
      Tags:
        - Key: Name
          Value: ASGInstance
          PropagateAtLaunch: true
```
### Outcomes
- Fully automated IaC deployment
- Real‑time monitoring and alerts
- Practical experience with serverless and DevOps concepts
<p><img width="986" height="1035" alt="image" src="https://github.com/user-attachments/assets/51e8f589-de38-4a16-9cf8-45a15e016641" /></p>

---

## Project: Deploy a Secure Web Application on AWS

### Overview
Deployed a production‑style web application using AWS services with layered security and monitoring.

### Architecture Components
- VPC with public/private subnets
- EC2 hosting Apache web server
- S3 with read‑only access
- RDS (MySQL) in private subnet
- CloudWatch dashboards and alerts

### Outcomes
- Secure, monitored web application
- Strong understanding of cloud security layering
- Demonstrated automation and infrastructure monitoring

---

## Database Systems

Experience designing and implementing **relational, NoSQL, graph, and time‑series databases** based on business needs.

---
<a id="project-ai-enhanced-database-ecosystem-aws"></a>
## Project: AI‑Enhanced Database Ecosystem (AWS)

### Overview
Designed a multi‑database ecosystem for a movie rental business to support **structured, unstructured, graph, and time‑series data**, while enabling AI integration.

### Databases Used
- **PostgreSQL (RDS):** transactional data
- **DynamoDB:** flexible movie metadata
- **Neptune:** relationship modeling
- **Timestream:** trend analysis
- **Elasticsearch:** advanced search

### Outcomes
- Scalable, cost‑efficient architecture
- Databases aligned to access patterns
- Clear justification of design choices

---

## Project: Time Series Database for DevOps Monitoring

### Overview
Designed and implemented an AWS **Timestream** database to monitor DevOps infrastructure and enable real‑time anomaly detection.

### Tools
| Tool | Use |
|---|---|
| AWS Timestream | Time‑series data storage |
| AWS CLI | Environment access |
| Python / Boto3 | Infrastructure deployment |

### Outcomes
- Demonstrated benefits over traditional RDBMS
- Real‑time and historical monitoring capabilities
- Scalable architecture for DevOps analytics

---

## Enterprise Architecture, Strategy, and Risk

---

## Project: Unified Multi‑Domain Enterprise Architecture (TOGAF + Zero Trust)

### Overview
Created a unified enterprise architecture for a multi‑domain organization (Healthcare, Fintech, E‑commerce) focused on security, scalability, and cost optimization.

### Frameworks
- TOGAF ADM
- Zero Trust Architecture

### Outcomes
- Risk prioritization and ROI mapping
- Cloud migration strategy
- Improved cybersecurity posture

---

## Project Management, Systems Development, and Risk

---

## Project: AI Risk Mitigation Plan for E‑commerce Platform

### Overview
Developed a formal risk management plan for integrating AI into a legacy e‑commerce platform using industry frameworks.

### Methodologies
- Probability‑impact risk matrix
- NIST AI Risk Management Framework
- Risk register development

### Outcomes
- Identified top three AI risks
- Defined mitigation strategies and ownership
- Continuous risk monitoring framework

---

## Machine Learning and Artificial Intelligence

---

## Project: LSTM Stock Price Prediction (TensorFlow)

### Overview
Built an LSTM model to predict next‑day stock prices using historical Yahoo Finance data.

### Outcomes
- **R²:** 0.73
- Identified lag and tuning opportunities
- Real‑world time‑series forecasting experience

---

## Project: Real‑Time Machine Learning Pipeline

### Overview
Created an end‑to‑end real‑time ML system using **Kafka**, **SGDRegressor**, and **Streamlit** for live stock price predictions.

### Outcomes
- Fully streaming ML pipeline
- Real‑time prediction dashboard
- Strong back‑end and front‑end integration

---

## Summary

This portfolio demonstrates:
- End‑to‑end ML model development
- Cloud architecture and DevOps practices
- Database design across multiple paradigms
- Enterprise architecture and risk management
- Real‑time data streaming and deployment
``
