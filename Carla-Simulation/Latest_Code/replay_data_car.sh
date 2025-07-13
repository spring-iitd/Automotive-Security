#!/bin/bash

log_file="./Logs_Replay/can_data_logs.log"

commands=()
timestamps=()

# Create directories if they don't exist
createFolders() {
    [ ! -d "./Logs_Replay" ] && mkdir ./Logs_Replay
    [ ! -d "./Graphs_Replay" ] && mkdir ./Graphs_Replay
}

clearTmpFiles() {
    rm -rf ./Logs_Replay/*.log
    rm -rf ./Graphs_Replay/*.png
}

runRepLogScript() {
    python3 ./replay_car.py
}

startCanDump() {
    candump -tz vcan0 > "$log_file" &
    sleep 1s
}

killCanDump() {
    pkill -9 candump
}

postProcessLogs() {
    sed -i 's/^ //' "$log_file"
    # python3 ./update_can_logs.py
    # python3 ./add_random_jitter.py
    # python3 ./arbitration.py
    # python3 ./plot_order_of_packets.py
}

main() {
    createFolders
    clearTmpFiles
    startCanDump
    runRepLogScript
    killCanDump
    postProcessLogs
}

main
