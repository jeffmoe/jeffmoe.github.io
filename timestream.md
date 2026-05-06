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
- [Project Link]

---
