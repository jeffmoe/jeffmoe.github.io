---
title: Develop and Analyze a GAN and Classifier
parent: AI, High Performance Computing, and Ethical Considerations
nav_order: 1
---

### Overview
Developed a **CNN classifier** and **GAN** using the MNIST dataset (70,000 handwritten digits). The classifier achieved high accuracy, while the GAN successfully generated lifelike synthetic digits. Training was optimized using **multi‑GPU parallelization** and **Automatic Mixed Precision (AMP)**.

### Methodologies
- CNN‑based digit classifier
- GAN architecture with generator and discriminator
- PyTorch `DataParallel`
- Mixed precision (AMP)
- Performance metrics and visual analysis

### Tools

| Tool | Use |
|:---|:---|
| PyTorch | Model development and training |
| Torchvision | Dataset loading |
| Matplotlib | Visualization |
| NumPy | Data processing |
| Scikit‑learn | Evaluation metrics |
| psutil / os | Resource monitoring |

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
- [Project Link](https://github.com/jeffmoe/jeffmoe.github.io/blob/main/Project%20Docs/GAN_with_Classifier.ipynb)

<p><img width="846" height="468" alt="image" src="https://github.com/user-attachments/assets/a5acc71a-0db1-4eca-8b15-7db6886c742d" /></p>
<p><img width="833" height="468" alt="image" src="https://github.com/user-attachments/assets/ff4cb7c2-ed8e-487c-94c1-757e64a7dae2" /></p>
<p><img width="1182" height="568" alt="image" src="https://github.com/user-attachments/assets/050ca4d9-431a-4401-bb69-e4ad4a9db870" /></p>
---
