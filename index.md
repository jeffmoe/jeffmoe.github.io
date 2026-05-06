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
|---           |---                             |
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

### Outcomes
- **R²:** 0.812
- **RMSE:** 9.85 minutes
- **MAE:** 7.30 minutes
- Identified prediction weaknesses with longer delivery times

---

## Project: Implementation of a Discriminative Model (NFL Position Classification)

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

### Outcomes
- **Accuracy:** 62.3%
- Strong performance for wide receivers
- Identified class imbalance as a key limitation

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

### Outcomes
- Fully automated IaC deployment
- Real‑time monitoring and alerts
- Practical experience with serverless and DevOps concepts

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
