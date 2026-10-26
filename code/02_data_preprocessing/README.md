# Data Preprocessing

## Overview

This directory contains a Bash script that automates the preprocessing pipeline for reaction data used by **BRMapper**.

The script prepares the reaction dataset, extracts atom-to-atom mapped reaction candidates, transforms reactions into Kekulé form, generates Imaginary Transition State (ITS) datasets, and converts the resulting datasets into TU format.

The preprocessing pipeline is designed to make the preparation of training and test data reproducible and straightforward.

A small dataset of ring reactions has been included here as a test example, which can be used to test the pipeline.

## Features

The script performs the following steps automatically:

1. **Input validation**  
   Checks whether the reaction data CSV and the ZIP archive containing AAM candidates exist and whether the ZIP archive is readable.

2. **AAM candidate extraction**  
   Extracts the contents of the provided ZIP archive into the output directory.

3. **Train/test split**  
   Splits the input reaction dataset into separate training and test datasets.

4. **Kekulé transformation**  
   Transforms the reactions in both datasets into Kekulé form.

5. **ITS dataset generation**  
   Generates ITS datasets and corresponding ground-truth files using the extracted AAM candidates.

6. **TU format conversion**  
   Converts the generated datasets into TU format for subsequent processing.

7. **Output organization**  
   Stores the generated files in a structured results directory.

## Project Structure

The expected project structure is:

```text
02_data_preprocessing/
├── prepross_env.yml
├── codes/
│   ├── split_train_testdata.py
│   ├── find_tautomeres_transform2_kekule.py
│   ├── generateITSfromCorrectAAMs.py
│   └── convert_dataset_BRMapper.py
│   └── data_utils.py
├── data/
│   ├── ringreactions_gt.csv
│   └── ringreactions_canidates.zip
├── preprocess.sh
└── README.md
```

## Usage
The Bash script assumes that it is executed from the `BRMapper/code/02_data_preprocessing/` directory because the Python scripts are referenced using relative paths such as `codes/split_train_testdata.py`.

If you use Conda, activate the environment containing the required dependencies:

```bash
conda env create -f prepross_env.yml
conda activate aamapper
```

Run this command once:

```bash
sh data_preprocessing.sh
```

The script interactively requests three paths:

| Input | Description |
|---|---|
| Reaction Data List (CSV) | Path to the input reaction dataset |
| Zipped AAM candidates file (ZIP) | Path to the archive containing AAM candidates |
| Output folder path | Directory in which all generated results will be stored |

Example input paths:

```text
Reaction Data List:
data/ringreactions_gt.csv

Zipped AAM candidates:
data/ringreactions_canidates.zip

Output folder:
results
```

The output directory can be specified as either a relative or an absolute path.

## Output Structure

For an output directory named `results`, the pipeline creates the following structure:

```text
results/
├── split_data/
│   ├── train.csv
│   └── test.csv
├── ground_truth/
│   ├── train_ground_truth.csv
│   └── test_ground_truth.csv
├── TUData/
│   ├── ...
│   └── ...
└── aam_candidates/
    └── ...
```

### Output directories

- **`split_data/`**  
  Contains the training and test reaction datasets. The reactions are subsequently transformed into Kekulé form.

- **`ground_truth/`**  
  Contains the ground-truth files generated for the training and test datasets.

- **`TUData/`**  
  Stores the datasets converted into TU format.

- **`aam_candidates/`**  
  Contains the extracted AAM candidates from the input ZIP archive.

The exact files generated inside `TUData/` and `aam_candidates/` depend on the behavior and input data of the corresponding scripts.

## Execution Order

The pipeline executes the processing steps in the following order:

```text
Input CSV + AAM candidates ZIP
              |
              v
      Validate input files
              |
              v
     Extract AAM candidates
              |
              v
      Split train/test data
              |
              v
    Transform to Kekulé form
              |
              v
       Generate ITS datasets
              |
              v
       Convert to TU format
              |
              v
         Results directory
```

## Error Handling

The script uses Bash's `set -e` option to stop execution when a command fails.

It also checks that:

- The input CSV file exists.
- The AAM candidates ZIP archive exists.
- The ZIP archive passes an integrity check.
- Required input paths are provided.

If an error occurs, the script terminates instead of continuing with subsequent processing steps.

