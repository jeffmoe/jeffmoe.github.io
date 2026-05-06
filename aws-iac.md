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
### Outcomes
- Fully automated IaC deployment
- Real‑time monitoring and alerts
- Practical experience with serverless and DevOps concepts
<p><img width="986" height="1035" alt="image" src="https://github.com/user-attachments/assets/51e8f589-de38-4a16-9cf8-45a15e016641" /></p>

---
