---
pretty_name: Aviation Dataset
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
  - split: validation
    path: validation.parquet
  - split: test
    path: test.parquet
---

# Aviation Dataset

## Dataset Description

This dataset contains aviation-related information collected from multiple publicly available datasets.

It covers different areas of aviation, including aircraft, airlines, airports, flight operations, aviation accidents, weather information, and aviation reports.

The dataset was cleaned, processed, and converted into readable text records for AI and Small Language Model (SLM) applications.

## Dataset Source

The dataset used in this project was collected from publicly available datasets on Kaggle. We downloaded the data and performed our own data cleaning, preprocessing, and preparation.

## Dataset Statistics

The dataset contains **1,533,723 records**.

| Split | Records | Percentage |
|---|---:|---:|
| Training | 1,226,943 | 80% |
| Validation | 153,367 | 10% |
| Testing | 153,413 | 10% |
| Total | 1,533,723 | 100% |

## Dataset Categories

The dataset includes the following categories:

- Accident
- Airline
- Airport
- Aircraft
- Operations
- Weather
- Aviation reports
- Other aviation information

## Dataset Structure

The dataset contains four columns:

| Column | Description |
|---|---|
| text | Aviation information converted into readable text |
| source | Name of the original dataset |
| record_type | Category of the aviation record |
| event_id | Unique identifier for each record |

## Data Preparation

The following steps were performed:

1. Collected aviation data from multiple CSV files.
2. Inspected and cleaned the datasets.
3. Handled missing and empty values.
4. Converted structured information into readable text.
5. Combined related aviation report attributes using ITEM_ID.
6. Removed duplicate records.
7. Divided the dataset into training, validation, and testing sets.
8. Saved the final datasets in Parquet format.

## File Format

All dataset splits are stored in Parquet format.

- train.parquet
- validation.parquet
- test.parquet

## Intended Use

This dataset is prepared for aviation-related data analysis, machine learning experiments, and future aviation-focused AI and SLM development.

## Limitations

- The dataset combines information from multiple sources.
- Data quality and completeness may vary between sources.
- Further preprocessing may be required depending on the intended application.
- The dataset should not be treated as an authoritative source for real-time aviation operations or safety decisions.