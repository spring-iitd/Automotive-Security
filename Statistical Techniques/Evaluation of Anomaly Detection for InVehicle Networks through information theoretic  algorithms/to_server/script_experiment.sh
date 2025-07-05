WINDOW_SIZES=(0.00001 0.001 0.01 0.05 0.1 0.2 0.5 1.0)

# Path to your data files
BENIGN_PATH="benign_data.csv"
ATTACK_PATH="DoS_dataset_transformed.csv"

# Loop over window sizes
for WS in "${WINDOW_SIZES[@]}"
do
    echo "Running with window_size = $WS"
    python script.py --benign_data_path "$BENIGN_PATH" --attack_data_path "$ATTACK_PATH" --window_size "$WS"
    echo "-----------------------------------------"
done