---
title: Unified Multi‑Domain Enterprise Architecture (TOGAF + Zero Trust)
parent: Enterprise Architecture, Strategy, and Risk
nav_order: 1
---

### Overview
Created a unified enterprise architecture for a multi‑domain organization (Healthcare, Fintech, E‑commerce) focused on security, scalability, and cost optimization.

### Frameworks
- TOGAF ADM
- Zero Trust Architecture

### Details
<p><img width="843" height="476" alt="image" src="https://github.com/user-attachments/assets/d6ba372c-4e83-40d6-ada0-c2272d069e30" /></p>
<p><img width="840" height="474" alt="image" src="https://github.com/user-attachments/assets/a20baf54-afab-4533-ac23-11b412bd607a" /></p>
<p><img width="842" height="476" alt="image" src="https://github.com/user-attachments/assets/e6fe7a9a-d22f-457a-a900-178030fe72cf" /></p>
<p><img width="841" height="475" alt="image" src="https://github.com/user-attachments/assets/dd000883-52a1-4fc2-ae35-6e3853bafb40" /></p>

### Data Diagrams
Heathcare:
<p><img width="1638" height="883" alt="image" src="https://github.com/user-attachments/assets/ef50b89a-03d9-4fab-bafc-5ec218525928" /></p>
Fintech:
<p><img width="1578" height="919" alt="image" src="https://github.com/user-attachments/assets/9d4adda8-2fa1-4391-9075-30cd13bcaea9" /></p>
E-Commerce:
<p><img width="1654" height="876" alt="image" src="https://github.com/user-attachments/assets/e15b008a-fba1-456b-9a23-b39effcd6f65" /></p>
Unified Approach:
<p><img width="1662" height="872" alt="image" src="https://github.com/user-attachments/assets/989c6809-e9fb-475a-8830-e227b6afe7bb" /></p>

### Example Product Integration
This example is for a wearable IoT device that collects healthcare data and sends it to a phone application via Bluetooth. Next, we want the phone app to send that data over the internet to the private server which hosts the database where the data is located. When we send data over the internet, we need to ensure that it is encrypted in transit on the way to the database and at rest in the database itself. This is also where AlphaCorp should develop the APIs to ensure that secure transfer of data. Next, we want another API to transfer the data from the private server into the public cloud server. Moving forward AlphaCorp should also consider creating a singular, scalable cloud database for their healthcare patient data instead of each product having its own data location. Lastly this data is transferred from the cloud network to the web application for the Doctor to utilize to assist with patient diagnosis and other care activities.
<p><img width="840" height="475" alt="image" src="https://github.com/user-attachments/assets/eeda583f-8bca-4c88-84d7-1b3651443b77" /></p>
Data Flow Diagram:
<p><img width="1566" height="923" alt="image" src="https://github.com/user-attachments/assets/1632565f-ecef-466d-8ac3-1a34315bcdae" /></p>

### Outcomes
- Risk prioritization and ROI mapping
- Cloud migration strategy
- Improved cybersecurity posture
- [Project Link]

---
