#!/bin/bash

# Define the start and end values for the range
start_value=23.5
end_value=23.5
step=0.00625
diff=0.004

# Generate lower values dynamically from 22.5 to 26.5 with a step of 0.1
lower_values=()
current_value=$start_value
while (( $(echo "$current_value <= $end_value" | bc -l) )); do
    lower_values+=("$current_value")
    current_value=$(echo "$current_value + $step" | bc)
done

# Define upper values (just add 0.5 to each corresponding lower value)
upper_values=()
for lower in "${lower_values[@]}"; do
    upper=$(echo "$lower + $diff" | bc)
    upper_values+=("$upper")
done

echo "Running generate_data_car..."
./generate_data_car.sh

# Iterate over the range values
for i in "${!lower_values[@]}"; do
    lower="${lower_values[$i]}"
    upper="${upper_values[$i]}"
    
    echo "Running for range ${lower} to ${upper}..."
    echo ""
    
    # Define new directory names
    logs_dir="./Logs_${lower}_1"
    graphs_dir="./Graphs_${lower}_1"
    
    # Create directories if they don't exist
    mkdir -p "$logs_dir" "$graphs_dir"

    # Modify log_gen.py using sed
    sed -i -E "s|if spoof_mode or \(current_time >= [0-9]+\.[0-9]+ and current_time <= [0-9]+\.[0-9]+\)|if spoof_mode or (current_time >= ${lower} and current_time <= ${upper})|g" spoof_attack_car.py
    
    # sed -e "s|./Logs_[0-9.]+/|${logs_dir}/|g" \
    #     -e "s|./Graphs_[0-9.]+/|${graphs_dir}/|g" spoof_attack_car.py
    # sed -i -E "s|./Logs_[0-9]+\.[0-9]+/|${logs_dir}/|g" spoof_attack_car.py
    # sed -i -E "s|./Graphs_[0-9]+\.[0-9]+/|${graphs_dir}/|g" spoof_attack_car.py
    # sed -i -E "s|\./Logs_[0-9]+\.[0-9]+(_[0-9]+\.[0-9]+)?/|${logs_dir}/|g" spoof_attack_car.py
    # sed -i -E "s|\./Graphs_[0-9]+\.[0-9]+(_[0-9]+\.[0-9]+)?/|${graphs_dir}/|g" spoof_attack_car.py
    sed -i -E "s|\./Logs_[0-9]+\.[0-9]+(_[0-9]+(\.[0-9]+)?)?/|${logs_dir}/|g" spoof_attack_car.py
    sed -i -E "s|\./Graphs_[0-9]+\.[0-9]+(_[0-9]+(\.[0-9]+)?)?/|${graphs_dir}/|g" spoof_attack_car.py

    # Modify generate_data_car.sh dynamically before execution
    sed -e "s|./Logs/|${logs_dir}/|g" \
        -e "s|./Graphs/|${graphs_dir}/|g" \
        -e "s|log_gen_agent.py|spoof_attack_car.py|g" generate_data_car.sh > tmp_run_cmd.sh



    chmod +x tmp_run_cmd.sh

    # Run the modified run_cmd.sh
    ./tmp_run_cmd.sh

    # Clean up temporary script
    rm tmp_run_cmd.sh
done
