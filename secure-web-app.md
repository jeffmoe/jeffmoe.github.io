---
title: Deploy a Secure Web Application on AWS
parent: Cloud Architecture and Infrastructure
nav_order: 2
---

### Overview
Deployed a production‑style web application using AWS services with layered security and monitoring.

### Architecture Components
- VPC (Virtual Private Cloud) with public/private subnets
  - Isolated virtual network in AWS
  - Host architecture in a custom network in AWS
- EC2 (Elastic Compute Cloud) hosting Apache web server
  - AWS virtual server that is scalable with auto scaling groups
  - Chose instance types with AMI (Amazon Machine Images)
- S3 (Simple Storage Service) with read‑only access
  - Versioning, encryption, region duplication
  - Used for backups and static web hosting
- RDS (Relational Database Service) in private subnet
  - MYSQL for this project
  - Supports many common database engines.
  - Automated backups for recovery and security
  - Mult-availability zone deployment for availability
- CloudWatch dashboards and alerts
  - Monitoring and Logs for AWS services
  - Create alerts based on custom metrics
  - Analyze logs for incident response and recovery
<p><img width="869" height="694" alt="image" src="https://github.com/user-attachments/assets/6f50bc6f-f7d2-4412-b246-deaa61534d00" /></p>

### VPC Configuration
**Initializing the VPC**
1. Log into the AWS Management Console.
2. Under Services select Networking and Content Delivery.
3. Scroll to the bottom and select VPC.
4. Once on the VPC dashboard, select Create VPC.
5. Select VPC only for now as we will configure additional settings later.
6. Name the VPC MyWebAppVPC.
7. Make sure IPv4 CIDR manual input is selected and input 10.0.0.0/16 for the CIDR block size.
8. Hit create VPC.
9. Your new VPC should show up under the default VPC in the dashboard.

**Configuring the Subnets**
1. Under the VPC menu, select subnets.
2. Then select create subnet on the dashboard.
3. Select the VPC we created in the previous step.
    - MyWebAppVPC 
4. Give the subnet a name.
5. Public_Subnet and Private_Subnet were the two created for this assignment.
6. Assign the Ipv4 CIDR block.
    - Public Subnet: 10.0.1.0/24 for EC2 and S3 access.
    - Private Subnet: 10.0.2.0/24 for RDS instance.
7. After configuring the first subnet, you can select add new subnet.
8. Click create subnet.
9. Your subnets will show up on the dashboard after creation.

**Creating and Connecting the Internet Gateway (IGW)**
1. Under the VPC dashboard select internet gateways.
2. Once there select create internet gateway.
3. Name your gateway and select create internet gateway.
4. Your gateway will show up in the dashboard, and the state will be detached.
5. Check the box next to the gateway name and under actions select connect to VPC.
6. Select the VPC from the dropdown and hit Attach Internet Gateway.
7. On the dashboard the state should now be attached, with the VPC ID listed.

**Configuring the Public Subnet Route Table**
1. Under the VPC dashboard chose Route Tables in the navigation pane.
2. Select create route table and give it a name.
3. Select the VPC you want from the dropdown and hit create route table.
4. After creation, go to actions and hit edit routes.
5. Under destination, set it to 0.0.0.0/0.
6. For Target, select Internet Gateway and select the gateway we created from the dropdown.
7. Hit save changes.
8. Go back to actions and click edit subnet associations.
9. Click the box next to your subnet and hit save associations.
10. Navigate back to your subnets and select the public subnet.
11. Under actions select edit route table association.
12. Change the route table from the default to the one just created from the dropdown and hit save.
13. You should now see your new route table listed on the configuration page for the subnet.

### EC2 Deployment
**Creating the Security Group**
For this assignment we are creating two groups, one for general security and another specifically for our database.
1. Under services go to the VPC dashboard.
2. Scroll down on the navigator to security groups.
3. Select create security group.
4. Name the group and give it a description.
5. Select what VPC you want it associated with and configure your inbound and outbound rules. 
    - SSH (Port 22) for secure access (only from your IP).
    - HTTP (Port 80) for web traffic.
6. When you are done hit create security group. 
7. You should be able to see your group in the AWS dashboard if successful.

**Creating the EC2**
1. Go to EC2 Dashboard under services menu.
2. Click launch instance.
3. Fill out the following information in the instance:
    - Chose an AMI – Amazon Linux 2
    - Chose an instance type – I chose a t2.micro that fell into the free tier
    - Pick number of instances, the VPC to host on, and subnets
    - Add storage – used default value here
4. Hit launch and you should see your EC2 on the dashboard.

**Connecting to the EC2 Instance and Deploying a Web Server**
1. Open a PowerShell window.
2. Type the following line of code:
```bash
Ssh –i /filepath/keyname.pem ece-user@publicip
```
You can grab the public IP from the EC2 configuration page.

3. Once connected type the following code to deploy the web sever:
``` bash
Sudo yum update –y
Sudo yum install httpd –y
Sudo systemctl  start httpd
Sudo systemctl enable httpd
Echo “Hello from My AWS Web App” | sudo tee /var/www/html/index.html
```
4. Open a new browser window and go to the ip address to see the message!

### S3 Configuration Steps
**Creating the Bucket**
1. Log into the AWS management console
2. Go to storage under services and select S3
3. Give the bucket a unique name
    - This cannot be the same as anything else in AWS
4. Select what region you want to store in
    - Mine is under the default US-East-1
5. Select whether you want the public to be able to access the bucket
6. Chose what encryption you want for your bucket
7. Chose server-side encryption with keys managed by S3 (SSE-S3)
8. You can enable or disable versioning on initial creation
9. When done hit create bucket
10. You should be able to see it on the dashboard screen

**Uploading an Object**
1. Select upload under your bucket.
2. Chose your file (image, text, etc.), name it, and select upload.
3. You will be able to see the object in the bucket dashboard after uploading.

**Allowing EC2 Read Access to the S3 Bucket**
1. Navigate to IAM under services.
2. Select role and then create role.
3. Chose AWS services and then EC2.
4. Select permissions and then attach the AmazonS3ReadOnlyAccess policy.
5. Name the role and create it.
6. Navigate to your EC2 under compute in services.
7. Select your EC2 and go to actions, security, and modify IAM role.
8. Select the role you created and save the changes.

### RDS Configuration
**Subnet Group Creation**
1. First, we need to create a subnet group for the database.
2. Under services, navigate to database and select Aurora and RDS.
3. On the left navigator, go down to subnet groups and then select create db subnet group.
4. Name your group and give it a description.
5. Chose the VPC you want it to go in and select your subnets then hit create.
6. Make sure that the subnets you select are hosted on different AZs.
7. Your subnet group should now show on the dashboard.

**Creating the Security Group**
1. For this assignment we are creating two groups, one for general security and another specifically for our database.
2. Under services go to the VPC dashboard.
3. Scroll down on the navigator to security groups.
4. Select create security group.
5. Name the group and give it a description.
6. Select what VPC you want it associated with and configure your inbound and outbound rules.
7. Allow access only from EC2 by using the private IP of the EC2.
8. When you are done hit create security group.
9. You should be able to see your group in the AWS dashboard if successful.

**Database Creation**
1. Under services, go to database and select Aurora and RDS.
2. In the navigation pane, select databases and then click create database.
3. Start off by choosing the database engine and template.
    - Engine: MySQL
    - Template: Free tier – single AZ deployment
4. Next is to configure the database credentials
5. You can enter in a username and a password or have AWS generate one for you
6. You will be able to store this password upon database creation
7. Next, we chose our instance type and storage needs
8. Then we chose which VPC to host the DB in, along with the subnet group we just created
9. Then we chose our security group that we created previously for the database
10. Next we select what AZ to host the DB in
11. Make sure is the same AZ that our private subnet is in, not the public one
12. Under additional configuration, make sure that the box for encryption is selected and that we use the default KMS key for the DB
13. Finally select create database.
14. You will see your DB in the dashboard upon successful creation.

**Credential Management**
1. Upon DB creation, click the banner at the top of the console to view the username and password.
2. Copy the password for now so it can be securely entered into Systems Manager Parameter Store.
3. Under services go to Management and Governance, then go to Systems Manager.
4. In the navigator pane, select parameter store and click create parameter.
5. Give your parameter a name and change its type to SecureString.
6. Then copy the value and hit create parameter.

### Cloud Watch Monitoring
**Creating the Dashboard**
1. Navigate to EC2 under compute in the services menu.
2. Copy your instance id for easier searching in the metric menu in CloudWatch.
3. Navigate to CloudWatch under Management and Governance in the services menu.
4. In the left-hand navigator scroll down to all metrics and paste your instance id into the search bar.
5. Select EC2>per instance and select CPU utilization.
6. At the top of the screen select actions and then add to dashboard.
7. Hit create new on the dashboard screen and name it.
8. Hit create and then add to dashboard. Your dashboard will show up under the dashboards section of CloudWatch.

**Creating the Alarm**
1. In the left-hand navigator, select alarm then create alarm.
2. Select metric and select EC2 and then per instance metric.
3. Paste your instance id again into the search bar and select CPU utilization and then select metric.
4. Change the statistic to sum and the threshold to greater than or equal to and enter 70.
5. Hit next and then select an existing SNS topic to send your email a notification when it goes into alarm.
6. Note that if you have not set this up already you will need to select create new topic instead.
7. Hit next, give your alarm a name, hit next again and then create alarm.

### Challenges and Observations
- Ensuring proper port access and security permissions to access and deploy a web server.
- Ensuring the database can only be accessed internally by the EC2 by using IAM roles.
- Creating the proper gateway connections so the EC2 can be accessed by the internet.
- Creating proper routes for route tables so public ips can’t access data on the private subnet.

### Outcomes
- Secure, monitored web application
- Strong understanding of cloud security layering
- Demonstrated automation and infrastructure monitoring

---
