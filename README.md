---
pretty_name: Airplane Crashes and Fatalities Since 1908
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

# Airplane Crashes and Fatalities Since 1908

## Dataset Description

This dataset contains historical information about airplane crashes
and aviation accidents recorded since 1908.

It includes information about accident dates, locations, operators,
aircraft details, people aboard, fatalities, ground fatalities,
and accident summaries.

## Dataset Structure

The dataset contains 13 columns:

| Column | Description |
|---|---|
| Date | Date of the accident |
| Time | Time of the accident |
| Location | Accident location |
| Operator | Airline or aircraft operator |
| Flight # | Flight number |
| Route | Flight route |
| Type | Aircraft type |
| Registration | Aircraft registration |
| cn/In | Aircraft construction number or identification |
| Aboard | Total people aboard |
| Fatalities | Number of fatalities aboard |
| Ground | Ground fatalities |
| Summary | Accident description |

## Dataset Splits

The dataset is divided into three parts:

| Split | Records | Percentage |
|---|---:|---:|
| Training | 3,687 | 70% |
| Validation | 790 | 15% |
| Testing | 791 | 15% |
| Total | 5,268 | 100% |

## Data Preparation

- The original CSV file was loaded and inspected.
- Empty values were identified as missing values.
- Numerical columns were converted to numerical data types.
- No duplicate records were found.
- The dataset was randomly divided into three parts.
- Random state 42 was used for reproducibility.
- All three datasets were saved in Parquet format.

## File Format

All three dataset files are stored in Parquet format.

## Intended Use

This dataset is prepared for future data analysis
and machine learning experiments.

The prediction target has not yet been selected.

## Limitations

Some columns contain missing values.
The dataset requires further preprocessing depending
on the machine learning task.

## Source Information

The original CSV file was supplied for this project.
The original external source and dataset license
have not yet been independently verified.
