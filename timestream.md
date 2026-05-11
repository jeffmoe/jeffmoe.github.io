---
title: Time Series Database for DevOps Monitoring
parent: Database Systems
nav_order: 2
---

### Overview
Designed and implemented an AWS **Timestream** database to monitor DevOps infrastructure and enable real‑time anomaly detection.

### Tools

| Tool | Use |
|---|---|
| AWS Timestream | Time‑series data storage |
| AWS CLI | Environment access |
| Python / Boto3 | Infrastructure deployment |

### Project Background
- Cloud-based technology solution company that is rapidly growing and supporting a wide variety of industries.
- Currently having internal performance issues with its DevOps monitoring as the large scale of time series data is putting strain on the existing database.

**Problem Statement**  
- Implement a Time-Series Database to effectively monitor the DevOps infrastructure for the company.

### Three Key Problem Areas
**Issues With Data Volume**
- DevOps Monitoring – The process of analyzing and monitoring network infrastructure and performance based on key performance indicators (KPIs) that are set by IT.
- Because the company is rapidly growing, they are supporting more and more infrastructure for many different applications.
- Monitoring this infrastructure to ensure high availability and performance for their users is causing a large volume of time series data to be generated.
- The volume of this data causes problems on traditional databases that do not have high write throughput.

**Issues with Real-Time Analysis**
- Because of the industries that the company supports with their cloud platform, such as monitoring climate and traffic patterns, high availability and performance are critical aspects.
- The Company needs to be able to see these variables in real-time to ensure that they are operating at the levels they need to.
- Their current setup struggles with real-time reporting because traditional databases store data on the disk, which causes latency in data retrieval.
- There are also complex queries required to access the data that is needed for the analysis needed, causing additional latency as the queries need to search entire DBs for the data requested at times.

**Issues with Decision Making**
- Because of the current setup, the company also has struggles with decision making.
- Relational databases are not designed specifically for time-series data, so additional data processing is required to get the data in a state that can be analyzed for effective decision making.
- These extra processing steps can lead to errors and inaccurate analysis.
- This makes things like anomaly detection and highlighting where performance issues started difficult, which are two critical variables in DevOps decisions.

### Devops Monitoring
- DevOps Monitoring – The process of analyzing and monitoring network infrastructure and performance based on key performance indicators (KPIs) that are set by IT.
- There are a few main goals of DevOps monitoring which are anomaly detection, performance monitoring, and capacity planning.
        - Performance Monitoring – process of monitoring the health of IT infrastructure in real-time. This would be things like CPU usage, latency, etc.
        - Anomaly Detection – process of looking at and finding unusual patterns, such as spikes in network traffic for an application or latency.
        - Capacity Planning – process of looking at historical data to drive and make business decisions on the allocation of infrastructure and resources.


### Time Series DBs
- Time-Series data is data where Time is a primary measurement. 
	- An example of this would be sensor data collected in 5 min intervals.
	- This data can be collected in regular or irregular intervals.
- Time-Series Databases (TSDBs) are DBs that are designed and optimized for time-series data.
- TSDBs excel and have architecture to support querying data over long time periods unlike traditional DBs.
- TSDBs have data expiration methods built in, to ensure that you don’t have database bloat as collect more and more data points.

**AWS Timestream**
- Fully managed database engine for time-series data on AWS.
- Provides scalability benefits being under the AWS umbrella.
- No need to worry about underlying architecture for deployment.
- Utilizes all the security features that AWS has to offer.
- Built on top of widely used open sourced DBs such as InfluxDB and Telegraf.

### Timestream Architecture
- Timestream takes in time series data from a source and stores it in one of two locations called storage tiers.
	- In-Memory tier – handles the processing of the data into the database and allows for keeping the data here for a set period. It is optimized to have minimal latency for real-time analytics.
	- Magnetic tier – was designed to support analytical queries and is low-cost long-term storage of data.
- Timestream also has a dedicated query layer for easy querying of data using a language that is very similar to SQL.
- Timestream stores data in tables, but you do not need to specify the columns upon table creation like you would in a traditional database.
	- These “columns” are called dimensions, and each table can support 128 unique values. 
	- The data type for all these dimensions is varchar.

### Timestream Integration
Step 1: Determine the data that needs to be collected  
- Key performance indicators from each piece of architecture that needs to be monitored.
    - Uptime – How often is a device sending and receiving data.
    - CPU Utilization – What percent of computer resources are being used on a device.
    - Throughput – Amount of traffic passing through a device.
    - Bandwith Usage - Amount of data that is being sent and received by a device.
  
Step 2: Choosing the right dimensions and measures from the existing database
- Need to ensure that the dimensions make sense and that they are consistent across the applications the company has deployed.  
    - Choosing the right dimensions will make querying and filtering easier and faster for analysis.  
    - Application_ID – specify which application the architecture is being used on.  
    - Application_Name – easier when pulling data for analytics for efficient queries and visualization.  
    - Time_Stamp – Need to pull a timestamp for time series data to be effective.  
    - CPU_Usage – This will determine the load on the application and could even trigger scaling policies if needed.  
    - HTTP_Status – Pulling this code will let you know if the application is running or not (Uptime).  
    - Network_Throughput – Value to let you know how much traffic is passing through an application.  
    - Memory_Usage – Percent of memory used in an application.  
  
Step 3: Choosing a partition key  
- For the company the partition key should be Application_Name so the data can be efficiently searched and visualized based on each IoT application that the company has deployed.

Step 4: Determine the data retention polices for storing data in memory for real-time analysis/ anomaly detection
- For monitoring network health, I think it makes sense to keep data in memory for low latency access for 8 hours at a time.
- After, it can then be sent to the more permanent magnetic storage for historical analysis.  

### Timestream Creation in AWS
1. Log into the AWS management console by creating an account.
2. Under services, navigate to Security, Identity, and Compliance and select IAM.
3. Once there, select users and create user.
4. Name the user and hit next until you create the user.
5. Once created select the user, go to the security credentials tab and select create access key.
6. Select the option for local code use.
7. Hit create and download the .csv file for safe keeping.

### Local Timestream Access
- Next on your local machine open your python environment to install Boto3 by using
```bash
  python –m pip install boto3
```
- Boto3 – Python API to perform CRUD operations in Timestream locally.
- Then we want to download and install the AWS command line interface (CLI) to modify our AWS configuration files easily.
- Once it has been downloaded and installed, open a CMD and type aws configure.  
            - It will prompt you to enter you access key, secret key, default region, and output format. 
- Enter in these values and you should be setup properly.  

### Using BOTO3 and CLI
The following code in your local instance will allow for database creation once you have successfully connected to AWS using your user access key:
```python
import boto3
timestream = boto3.client('timestream-write')
database_name = 'OmptimaTechDB'
try:
    response = timestream.create_database(DatabaseName=database_name)
    print(f"Database '{database_name}' created successfully.")
except timestream.exceptions.ConflictException:
    print(f"Database '{database_name}' already exists.")
except Exception as e:
    print(f"Error creating database: {e}")
table_name = 'DevOPsMetrics'
try:
    response = timestream.create_table(
        DatabaseName=database_name,
        TableName=table_name
    )
    print(f"Table '{table_name}' created successfully in database '{database_name}'.")
except timestream.exceptions.ConflictException:
    print(f"Table '{table_name}' already exists in database '{database_name}'.")
except Exception as e:
    print(f"Error creating table: {e}")
```
The following code was used in python to upload a test data set I created in a csv file to test writing data into Timestream:
```python
with open(csv_file_path, 'r', encoding='utf-8-sig', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    print("CSV Headers:", reader.fieldnames)
    for row in reader:
        dimensions = [
            {'Name': 'Application_ID', 'Value': row['Application_ID']},
            {'Name': 'Application_Name', 'Value': row['Application_Name']}
        ]
        metrics = ['CPU_Usage', 'HTTP_Status', 'Network_Throughput', 'Memory_Usage']
        records = []
        for metric in metrics:
            value_type = 'DOUBLE' if metric != 'HTTP_Status' else 'BIGINT'
            records.append({
                'Dimensions': dimensions,
                'MeasureName': metric,
                'MeasureValue': row[metric],
                'MeasureValueType': value_type,
                'Time': parse_timestamp(row['Time_Stamp']),
                'TimeUnit': 'MILLISECONDS'
            })
try:
            timestream.write_records(
                DatabaseName=database_name,
                TableName=table_name,
                Records=records
            )
            print(f"Uploaded metrics for {row['Application_ID']} at {row['Time_Stamp']}")
        except Exception as e:
            print(f"Error uploading record: {e}")
            print(e.response)
            time.sleep(1)
```
Query by Application_Name:
```bash
aws timestream-query query --query-string 'SELECT * FROM "OmptimaTechDB"."DevOPsMetrics" WHERE Application_Name = ''TrafficManagementSystem'' ORDER BY time DESC LIMIT 100’  
aws timestream-query query --query-string 'SELECT Application_Name, time, measure_value::double FROM "OmptimaTechDB"."DevOPsMetrics" WHERE measure_name = ''CPU_Usage'' ORDER BY Application_Name, time’
```
Query by Time:
```bash
aws timestream-query query --query-string 'SELECT * FROM "OmptimaTechDB"."DevOPsMetrics" WHERE time < ago(1h)’  
aws timestream-query query --query-string 'SELECT * FROM "OmptimaTechDB"."DevOPsMetrics" WHERE time BETWEEN ago(10m) AND ago(5m)’
```
### Outcomes
- Demonstrated benefits over traditional RDBMS
- Real‑time and historical monitoring capabilities
- Scalable architecture for DevOps analytics
- [Project Link](https://github.com/jeffmoe/jeffmoe.github.io/blob/main/Project%20Docs/Assessing%20Time%20Series%20Databases%20in%20DevOps%20with%20OptimaTech.pptx)

---
