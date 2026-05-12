---
title: Power Bi Dashboard for Sales Program
parent: Professional Experience
nav_order: 2
---

## Overview
This project delivers a **Power BI dashboard** that tracks and visualizes participation in a sales incentive program based on user submissions.

The solution automates data collection, transformation, and reporting using the Microsoft Power Platform.

---

## Architecture & Data Flow

1. **Microsoft Forms**
   - Users submit participation data through a structured form.

2. **Power Automate**
   - A workflow triggers on each submission.
   - The flow processes the response and writes the data into a SharePoint list.

3. **SharePoint List**
   - Acts as the centralized data source.
   - Stores structured submission records.

4. **Power BI**
   - Connects directly to the SharePoint list.
   - Provides real-time dashboards and analytics for program tracking.

---

### Dashboard 1: Program Performance Overview

<p><img width="1479" height="754" alt="image" src="https://github.com/user-attachments/assets/27ae79ac-7c05-4512-a4ad-164cdcd36fa3" /></p>

### Key Visual Components

- **Category Points (Top Left)**
  - Displays points grouped by region
  - Segmented into:
    - Tier 1 (light blue)
    - Tier 2 (dark blue)

- **Quote and Order Points (Middle)**
  - Comparison of quote points vs order points by region
  - Highlights performance pipeline from quoting to ordering

- **Leaderboard (Right Panel)**
  - Ranks participants based on total points
  - Includes slicers:
    - ISE filter
    - Region filter

- **Category Breakdown (Middle Lower Section)**
  - Categories include:
    - Pressure
    - Level
    - Temperature
    - Wireless
  - Each broken down into Tier 1 and Tier 2 contributions

- **KPIs (Bottom Section)**
  - Total Submissions: 4717
  - Expedites: 1396
  - Tier 1 Points: 23,133
  - Tier 2 Points: 39,846
  - Expedite Points: 9,772
  - Total Points: 72,707

---

### Dashboard 2: Submission Details

<p><img width="1477" height="754" alt="image" src="https://github.com/user-attachments/assets/4f1ce94c-42bb-47ec-8cbb-1af1bab33192" /></p>

### Key Visual Components

#### Top Section
- **Quote Submissions Chart**
  - Distribution of quote submissions by participant
- **Order Submissions Chart**
  - Distribution of completed orders

- **Date Filters**
  - Adjustable timeline (e.g., 01/01/2025 – 07/29/2025)

---

#### Filtering Controls
- Manager
- Region
- ISE (Inside Sales Engineer)
- Order Number / Quote Number
- Category filters

---

#### Middle Section

- **Quote Points by User**
  - Tier 1 vs Tier 2 breakdown per participant

- **Order Points by User**
  - Highlights conversion and fulfillment performance

---

#### Bottom Section (Tables)

- **Quotes Table**
  - Columns:
    - ISE
    - Quote Number
    - Quote Date
    - Quote Categories

- **Orders Table**
  - Columns:
    - ISE
    - Quote Number
    - Order Number
    - Order Categories
---

## Key Insights Enabled

- Participation tracking across regions and individuals
- Conversion visibility (quotes → orders)
- Performance benchmarking via leaderboard
- Category-level contribution analysis
- Identification of high-performing segments and trends

---

## Value of the Solution

- Fully automated data pipeline
- Real-time reporting and insights
- Scalable and maintainable architecture
- Improved visibility into sales engagement
- Enables data-driven incentives and decision-making

---

## Future Enhancements

- Role-based access control (RLS in Power BI)
- Advanced trend forecasting
- Integration with CRM or ERP systems
- Enhanced anonymization layer for external sharing

---

``
