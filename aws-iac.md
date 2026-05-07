---
title: AWS IaC Deployment with Serverless Alerting
parent: Cloud Architecture and Infrastructure
nav_order: 1
---

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
### Deployment Steps
**Grab AMI image id**
1. Navigate to Services and then Compute
2. From compute select EC2 and in the left navigator in the dashboard scroll down to AMI Catalog
3. Search for Amazon Linux 2 AMI and copy id for template
   
**Grab Subnet Id**
1. Navigate to Services and then Networking and Content Delivery
2. From networking and content delivery select VPC.
3. In the left navigator in the dashboard scroll down to Subnets
4. Copy the subnet id from the dashboard for us-east-a1 for template
   
**Deploy Template**
1. Navigate to Services and then Management and Governance
2. From Management and Governance select CloudFormation
3. Select Create stack and then build from infrastructure composer
4. Select template at the top of the screen and enter in your template
   
### Lambda Function and EventBridge Setup
```python
import json
import boto3
def lambda_handler(event, context):
 message = f"EC2 Instance {event['detail']['instance-id']} has been terminated."
 print(message)
 return {
  'statusCode': 200,
  'body': json.dumps(message)
 }
```
**Setup**
1. Go to services and then compute
2. Select lambda and hit create function
3. Select Author from scratch and give your function a name and chose the runtime
4. Chose your permissions and additional configurations and hit create function
5. Scroll down to code source and enter your function code
6. Hit deploy to deploy it to AWS. The function should display on the dashboard upon successful completion

### Logs and Alarms
**Steps**
1. Go to services and select Management and Governance
2. Select CloudWatch and in the left-hand navigator go down to logs and then log groups. Verify the log group appears for the Lambda function we created
3. Click on the log group for the function and select the metric filters tab
4. Select create metric filter and enter EC2 Instance terminated for the filter pattern
5. Select next and name your filter and namespace if you do not have one created already
6. Give your metric a name and value to generate when triggered and hit next and hit create metric
7. You will see your metric filter populated now for the log group
8. Check the box for the metric and then hit create alarm
9. Change the period down to 1 minute and update the conditions to be greater than or equal to 1
10. Hit next and select either an existing SNS type or create a new one if your email is not setup already
11. If you need to setup a new one you will receive an email to verify the address before you can receive alarm emails
12. Hit next, give your alarm a name, hit next again and then create alarm
13. You will see your alarm in the dashboard upon successful creation

### Security Considerations
- Normally you would not want any IP address to have access to parts of your infrastructure
- When creating a stack, using IAM roles to limit who can update and change it is important for security
- Storing values such as subnet Ids using parameter store provides an additional layer of security
- Creating notifications for changes is important to see if anyone is changing infrastructure without approval

### Outcomes
- Fully automated IaC deployment
- Real‑time monitoring and alerts
- Practical experience with serverless and DevOps concepts
<p><img width="986" height="1035" alt="image" src="https://github.com/user-attachments/assets/51e8f589-de38-4a16-9cf8-45a15e016641" /></p>

---
