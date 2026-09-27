#!/bin/bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: bash run_kaggle.sh <PATH_TO_DATASET>"
    echo "Example: bash run_kaggle.sh /kaggle/input/amazon-ml-dataset/dataset"
    echo ""
    echo "Expected Directory Structure inside <PATH_TO_DATASET>:"
    echo "  ├── train/"
    echo "  │   ├── train_source1.tsv"
    echo "  │   ├── train_source2.tsv"
    echo "  │   ├── train_source3.tsv"
    echo "  │   └── train_ground_truth.tsv"
    echo "  └── test/"
    echo "      ├── test_source1.tsv"
    echo "      ├── test_source2.tsv"
    echo "      └── test_source3.tsv"
    exit 1
fi

DATA_DIR=$1
TRAIN_DATA="$DATA_DIR/train"
TEST_DATA="$DATA_DIR/test"

echo "======================================================"
echo "    Starting cascadER Kaggle Pipeline (Maximized)     "
echo "======================================================"

echo "Installing Requirements..."
pip install -r requirements.txt

mkdir -p artifacts
mkdir -p output

# 1. Build training indexes and base features
echo "1. Building indexes and base features..."
# Note: hash-start 0 and hash-stop 10000 uses 100% of the training data instead of a tiny sample,
# maximizing the XGBoost model's F-score capability.
python src/sample_anchors.py --train-dir "$TRAIN_DATA" --out artifacts/sample --hash-start 0 --hash-stop 10000 

python src/pipeline.py index --sources "$TRAIN_DATA/train_source2.tsv" "$TRAIN_DATA/train_source3.tsv" --db artifacts/train.sqlite
python src/retrieval_extras.py --db artifacts/train.sqlite --out artifacts/train.sqlite.extra.sqlite
python src/reference_index.py --source "$TRAIN_DATA/train_source1.tsv" --out artifacts/references.sqlite
python src/precompute_competition.py --db artifacts/train.sqlite --references artifacts/references.sqlite --out artifacts/competition --workers 4

# Top-k increased to 120 (from 60) for maximum candidate recall on Kaggle's 30GB RAM
python src/cache_training.py --data artifacts/sample --db artifacts/train.sqlite --out artifacts/base --top-k 120 --max-block 1500 --workers 4

# 2. First stage and contextual features
echo "2. Running First Stage Model and Context Features..."
python src/crossfit_scores.py --cache artifacts/base --out artifacts/crossfit --trees 800 --depth 7
python src/build_eid_lookup.py --db artifacts/train.sqlite --out artifacts/lookup.sqlite
python src/augment_parallel.py --cache artifacts/base --lookup artifacts/lookup.sqlite --competition artifacts/competition --posterior-dir artifacts/crossfit --out artifacts/context --workers 4

echo "3. Building Robust TF-IDF Text Weights..."
python src/robust_features.py --db artifacts/train.sqlite --out artifacts/text_weights.joblib
python src/augment_robust.py --cache artifacts/context --lookup artifacts/lookup.sqlite --weights artifacts/text_weights.joblib --out artifacts/robust --workers 4

# 3. Fit, select, and evaluate
echo "4. Fitting Stage 2 XGBoost Model..."
python src/fit_cached.py --cache artifacts/robust --out artifacts/final-model --experiments 2000:7,2500:6
python src/evaluate_cached_model.py --cache artifacts/robust --model artifacts/final-model

# 4. Generate Final Predictions on Test Data
echo "5. Precomputing Test Indexes..."
python src/pipeline.py index --sources "$TEST_DATA/test_source2.tsv" "$TEST_DATA/test_source3.tsv" --db artifacts/test.sqlite
python src/retrieval_extras.py --db artifacts/test.sqlite --out artifacts/test.sqlite.extra.sqlite
python src/reference_index.py --source "$TEST_DATA/test_source1.tsv" --out artifacts/test_references.sqlite
python src/precompute_competition.py --db artifacts/test.sqlite --references artifacts/test_references.sqlite --out artifacts/test_competition --workers 4

echo "6. Generating Final Test Predictions..."
python src/predict_parallel.py --source1 "$TEST_DATA/test_source1.tsv" --db artifacts/test.sqlite --competition-cache artifacts/test_competition --model artifacts/final-model --out output --workers 4

echo "======================================================"
echo "    Pipeline Complete! Output is in ./output/         "
echo "    You can submit matching_results.tsv and           "
echo "    candidate_pairs.tsv directly to the leaderboard!  "
echo "======================================================"
