---
title: AI‑Enhanced Database Ecosystem (AWS)
parent: Database Systems
nav_order: 1
---

### Overview
Designed a multi‑database ecosystem for a movie rental business to support **structured, unstructured, graph, and time‑series data**, while enabling AI integration.

### Databases Used
- **PostgreSQL (RDS):** transactional data
- **DynamoDB:** flexible movie metadata
- **Neptune:** relationship modeling
- **Timestream:** trend analysis
- **Elasticsearch:** advanced search
 
### Project Diagrams and Info
**Architecture Diagram**
<p><img width="1129" height="634" alt="image" src="https://github.com/user-attachments/assets/c4ef9a9c-f28f-4dda-9677-ed1b06047670" /></p>

**Security Diagram**
<p><img width="1035" height="648" alt="image" src="https://github.com/user-attachments/assets/e85655d1-f154-4408-bcfc-7447c0dcd4ab" /></p>

**Entity Relationship Diagram**
<p><img width="1210" height="614" alt="image" src="https://github.com/user-attachments/assets/8b5a6bd3-b0b0-4090-93b1-bf2e0cca0972" /></p>

**Architecture with AI Integration**
<p><img width="978" height="641" alt="image" src="https://github.com/user-attachments/assets/d4e36e95-6620-48fd-8510-82fb2e90fc90" /></p>

### Example Queries
Postgre: This query shows all the movies that were rented more than 30 times last month. This allows the company to adjust their inventory appropriately for what is being rented.
```sql
SELECT m.MovieID, m.Title, COUNT(r.RentalID) AS rentals FROM Movies m JOIN Rentals r ON m.MovieID = r.MovieID WHERE r.RentalDate >= DATE_TRUNC('month', CURRENT_DATE INTERVAL '1 month') GROUP BY m.MovieID HAVING COUNT(r.RentalID) > 30 ORDER BY rentals DESC;
```
DynamoDB: This code and query graphs all the movies that were added to the dynamodb database in the past three months. 
```python
import boto3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dynamodb_json import json_util

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Movies')  # Replace with your table name

three_months_ago = (datetime.now() - timedelta(days=90)).isoformat()

response = table.scan(
    FilterExpression='added_date >= :date',
    ExpressionAttributeValues={
        ':date': three_months_ago
    }
)
movies = json_util.loads(response['Items'])
movie_titles = []
add_dates = []
genres = []
for movie in movies:
    movie_titles.append(movie.get('title', 'N/A'))
    add_dates.append(datetime.fromisoformat(movie['added_date']))
    genres.append(', '.join(movie.get('genres', [])))

plt.figure(figsize=(12, 8))
plt.plot_date(add_dates, movie_titles, linestyle='none')
plt.title('Movies Added in Last 3 Months')
plt.ylabel('Movie Title')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
Neptune: Here is a specific query to see which genre has the most rentals.
```gremlin
g.V().hasLabel('Genre').project('genre','rental_count').by('name').by(in_('CONTAINS').count())order().by('rental_count', decr).toList()
```
Timestream: This query is more advanced, as it shows the breakdown of movie rentals per hour. 
```sql
SELECT bin("rented timestamp", 1h) AS hour_bin, COUNT(*) AS rentals FROM "movie-rental-database"."Rentals" WHERE "Rented" BETWEEN ago(90d) AND now() GROUP BY bin("Rented", 1h) ORDER BY hour_bin;
```
### Outcomes
- Scalable, cost‑efficient architecture
- Databases aligned to access patterns
- Clear justification of design choices
- [Project Link]
---
