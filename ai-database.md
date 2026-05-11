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
**High Level Overview**
<p><img width="1034" height="634" alt="image" src="https://github.com/user-attachments/assets/e278d1e9-9d93-4c7a-bddc-2a64e0f6b6b9" /></p>

**Architecture Diagram**
<p><img width="1129" height="634" alt="image" src="https://github.com/user-attachments/assets/c4ef9a9c-f28f-4dda-9677-ed1b06047670" /></p>

**Security Diagram**
<p><img width="1035" height="648" alt="image" src="https://github.com/user-attachments/assets/e85655d1-f154-4408-bcfc-7447c0dcd4ab" /></p>

**Entity Relationship Diagram**
<p><img width="1210" height="614" alt="image" src="https://github.com/user-attachments/assets/8b5a6bd3-b0b0-4090-93b1-bf2e0cca0972" /></p>

**Architecture with AI Integration**
<p><img width="978" height="641" alt="image" src="https://github.com/user-attachments/assets/d4e36e95-6620-48fd-8510-82fb2e90fc90" /></p>

### Problem Statement
- The Movie Rental Company has a vast collection of movies and users of different demographics with changing preferences.
- The Movie Rental Company wants to implement a modern and AI driven data management ecosystem to create a personalized movie recommendation system in an effort to increase customer retention on the platform.
- The goal is to store data using the appropriate database management system type so it can be properly used by AI algorithms to make better business decisions, provide insight into inventory, and provide an overall better user experience

### Specific Problem Areas
- Vast Movie Library: The rental company has a wide variety of movies new and old that is constantly being added to and updated. This data needs to be organized efficiently so it can be accessed for user recommendations.
- Growing Customer Information: The rental company has many users of its platform and this is constantly growing. The data includes demographics information along with information about their memberships and rental history.
- Changing Customer Preferences: The company hopes to stay on top of trends and changing customer preferences by better utilizing the data the company has on its users and movie library to have more informed business decisions on things like content and inventory management. 

### AI Integration
The Company hopes to integrate AI into its Database Management in four key ways.
- Automated Data Management: Company wants to automatically keep and make changes to records and categorize their vast movie backlog.
- Personal Customer experiences: create algorithms to provide analysis on rental behavior to have recommendations based on the specific user.
- Inventory Optimization: Company wants to use AI to predict preferences and trends to help ensure that the movies people want are always available.
- Business Intelligence: Company wants to look at historical data to improve its targeted marketing and decision making. 

### DB Selection and Information
**Vast Movie Library**
- The movie library has so many different data types and structures.
- It also is constantly changing and updating as new items get added.
- Because of the varied data structure and schema changes a key value nosql database will work best for this section of the data.
- This flexible schema will allow us to continually update the database with new movies and genres without causing too much trouble.
- This also allows us to index the library better for faster queries which will be critical for searching, recommending, and AI integration.

**Growing Customer Information**
- Customer data is much more static than the library of movie data.
- Additional data integrity is needed for this for the actual transactions of renting movies.
- We have well defined relationships between customers, rentals, and payments.
- Because of these reasons a traditional relational database will be used still for this portion of the data.
- This also helps keep costs and complexity down as its less data to migrate into a new system. 

**Changing Customer Preferences**
- Because of the volume of users and movies, it can be hard to create queries and model relationships effectively with a traditional database.
- This would require the creation of many tables with different primary and foreign keys, and even then, there is work needed to understand the relationships fully.
- For this portion of data, we will be using a graph database to visualize and simplify the modeling of these relationships.
- Will help us identify similar customers who rented the same movie to power our personalized recommendations.

**Updated UX**
- On top of our three “main” data portions, we want to improve the user experience.
- Since we are migrating our movie data to a nosql database, we will need an improved and efficient way for customers to interact with it.
- Because of this the company should also implement a search engine database to be integrated with the nosql database.
- This allows for full text search capabilities and can be built with similar indexes to the nosql database for fast retrieval of movies.

**Business Intelligence**
 - Now that we have a way to simply model the relationships between the data, we need a way to easily analyze trends in rental patterns and customer behavior.
 - To do this, the company should implement a time series database.
 - This allows easy visualization of temporal data, so you can see things like the last time a customer rented a movie, or if a movie is being rented more frequently than it used to.
 - This will help the company keep on top of trends and make more informed business decisions.

**Utilizing the Cloud**
 - Using AWS, we can host all of these databases in the same cloud network.
 - This simplifies the connection process as amazon services have native integration with each other.
 - AWS DynamoDB for the Nosql dB, AWS RDS for the relational DB, AWS Neptune for the graph DB, AWS Timestream for the time series DB, AWS Elasticsearch for the search engine db.
 - We then can utilize Amazon’s native Identity Access Management tools as well to provide set credentials to our AI tools.
     - With IAM, we can set the permissions for the algorithms so they can only interact and change certain things. 
     - We will be following the principle of least access for this.

### DB Security
We will be adding security to our ecosystem in 5 key ways to ensure we have total security coverage and minimizing the chance for leaks or theft.
- Private subnets – we can store the actual data in a private subnet in our cloud network away from public access.
- Identity Access Management – we can use access controls and Multifactor authentication for user access as well as giving granular control to our AI models.
- Encryption – we will use secure socket layer and transit socket layer protocols to encrypt the data at rest and in transit.
- Security groups – we can further secure our network by creating security groups to direct traffic and create specific inbound and outbound rules for our databases.
- Tokenization – we can use the concept of tokenization to replace sensitive customer data (ie. Credit card info) in the database so even if data is queried it needs to be decrypted to see that information.

### DB Administration
**Administration - RDS**  
For our RDS, we will need to have 6 entities that should cover everything we need:
- Customers, payments, invoices, rentals, reviews, and movies
 
***Relationships***  
- Each customer can have multiple payment methods, and each payment method is assigned to one customer.
- Each customer can have multiple invoices, and each invoice is assigned to one customer.
- Each customer can have multiple rentals, and each rental is assigned to one customer.
- Each customer can leave multiple reviews, and each review is created by one customer.
- Each review is assigned to one movie. Each movie can have multiple reviews.
- Each invoice is assigned to one rental.
- Each movie can be rented multiple times; each rental is assigned to a movie.

**Administration - DynamoDB**  
Being a nosql database a schema does not really need to be defined at the time of database creation.
- We want to ensure that the data we store in the DB is what we need for quick queries.
- We also need to be able to perform analytics and have AI integration.
- To do this we will create a single table in DynamoDB called Movies.
- It will have a partition key of MovieId and a sort key of MovieTitle.
- Attributes in the table:
    - Director, genre, year, cast
- It will have two global secondary indexes for advanced queries.
    - One will use year as a value and the other will use director.

**Administration - Neptune**  
- For our graph database, we need to pull the data from the RDS and DynamoDB to model the relationships that we want to visualize. 
- AWS has built in integration tools to read data from those sources.
     - AWS DMS – data management service to automatically load data from other databases into Neptune. 
- Our vertices (entities) for our graph are customers, rentals, movies, and genre.
- We want to see how the company’s customers interact with the movies and how often movies are being rented.

**Administration - Timestream**  
- For our time series database, we can again use AWS DMS to migrate the data we need into Timestream.
- We are going to create two timestream tables, one for customer information and one for rental information.
- Rental table: RentalID, rented timestamp, time rented (Days), customerid, Customer Name, Movie Title, genre.
- Customer table: CustomerID, Customer name, membership level, number of rentals, last login timestamp.

### AI Integration 
- Being hosted on AWS, there are several AI tools that can be utilized to perform the automation and analysis that the company is looking for.
- Automation – AWS Lambda: serverless computing tool which can be used to automatically update our database information that we need for things like inventory management.
- Customer Recommendations – AWS Personalize: ML algorithm that creates recommendations based on the company data. Has integration with RDS and Neptune.
- Trend and other sales forecasting – AWS Sagemaker canvas: has the ability to integrate ML models to the timestream DB to perform forecasting on time series data.

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
- [Project Link](https://github.com/jeffmoe/jeffmoe.github.io/blob/main/Project%20Docs/Designing%20an%20AI-Enhanced%20Database%20Ecosystem%20for%20a%20Movie%20Rental%20Company.pptx)
---
