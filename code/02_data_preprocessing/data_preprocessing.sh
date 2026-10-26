#!/bin/bash

# Abort in the event of an error
set -e

# ==========================================
# Function for inputting a path
# ==========================================

read_path() {
    local prompt="$1"
    local variable_name="$2"
    local input_path

    read -r -p "$prompt: " input_path

    if [[ -z "$input_path" ]]; then
        echo "Error: No path has been entered." >&2
        exit 1
    fi

    printf -v "$variable_name" '%s' "$input_path"
}

# ==========================================
# Configuration
# ==========================================

echo "=========================================="
echo "       Configuration of the data set"
echo "=========================================="
echo

# Enter the input paths
read_path "Please insert the Reaction Data List (CSV)" rawdata_rxnlist
read_path "Please insert the zipped AAM candidates file (ZIP)" suboptimals_zip

# Enter the results folder
read_path "Please insert the output folder path" results_dir

# ==========================================
# Check input files
# ==========================================

if [[ ! -f "$rawdata_rxnlist" ]]; then
    echo "Error: Reaction Data List not found: $rawdata_rxnlist" >&2
    exit 1
fi

if [[ ! -f "$suboptimals_zip" ]]; then
    echo "Error: ZIP file not found: $suboptimals_zip" >&2
    exit 1
fi

if ! unzip -tq "$suboptimals_zip"; then
    echo "Error: ZIP file is invalid or cannot be read." >&2
    exit 1
fi

# ==========================================
# Define output paths
# ==========================================

train_data="$results_dir/split_data/train.csv"
test_data="$results_dir/split_data/test.csv"

train_ground_truth="$results_dir/ground_truth/train_ground_truth.csv"
test_ground_truth="$results_dir/ground_truth/test_ground_truth.csv"

path_TUdata="$results_dir/TUData"

train_folder_TUdata="train_TUdata"
test_folder_TUdata="test_TUdata"

# Folder for extracted AAM candidates
aam_candidates="$results_dir/aam_candidates"

mkdir -p "$results_dir/split_data"
mkdir -p "$results_dir/ground_truth"
mkdir -p "$results_dir/TUData"
mkdir -p "$aam_candidates"

echo
echo "+++++++++++ Extract AAM candidates +++++++++++"

unzip -o "$suboptimals_zip" -d "$aam_candidates"

echo "Extracted to: $aam_candidates"

echo
echo "+++++++++++ Split training and test datasets +++++++++++"
python codes/split_train_testdata.py \
    "$rawdata_rxnlist" \
    "$train_data" \
    "$test_data"

echo
echo "+++++++++++ Transform reactions to Kekule form +++++++++++"

python codes/find_tautomeres_transform2_kekule.py \
    "$train_data" "$train_data"

python codes/find_tautomeres_transform2_kekule.py \
    "$test_data" "$test_data"

echo
echo "+++++++++++ Generate ITS datasets +++++++++++"

python codes/generateITSfromCorrectAAMs.py \
    "$test_data" "$aam_candidates" "$test_ground_truth"

python codes/generateITSfromCorrectAAMs.py \
    "$train_data" "$aam_candidates" "$train_ground_truth"

echo
echo "+++++++++++ Convert datasets to TU format +++++++++++"

python codes/convert_dataset_BRMapper.py \
    "$test_folder_TUdata" "$path_TUdata" \
    "$aam_candidates" "$test_ground_truth"

python codes/convert_dataset_BRMapper.py \
    "$train_folder_TUdata" "$path_TUdata" \
    "$aam_candidates" "$train_ground_truth"

echo
echo "=========================================="
echo "All processing steps have been completed."
echo "Results are stored in: $results_dir"
echo "=========================================="

