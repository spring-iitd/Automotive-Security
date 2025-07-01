#!/bin/bash

gen_log_file="./Logs/can_data_logs.log"

commands=()
timestamps=()

# Create directories if they don't exist
createFolders() {
    [ ! -d "./Logs" ] && mkdir ./Logs
    [ ! -d "./Graphs" ] && mkdir ./Graphs
}

clearTmpFiles() {
    rm -rf ./Logs/*
    rm -rf ./Graphs/*.png
}

runGenLogScript() {
    python3 ./real_time_car.py
}

startCanDump() {
    candump -tz vcan0 > "$gen_log_file" &
    sleep 5s
}

killCanDump() {
    pkill -9 candump
}

postProcessLogs() {
    sed -i 's/^ //' "$gen_log_file"
    # python3 plot_order_of_packets.py
    # python3 arbitration.py
    # python3 merge_can_logs.py
    # python3 ./update_can_logs.py
}

main() {
    createFolders
    clearTmpFiles

    # Generate log data
    startCanDump
    runGenLogScript
    killCanDump
    postProcessLogs

}

main