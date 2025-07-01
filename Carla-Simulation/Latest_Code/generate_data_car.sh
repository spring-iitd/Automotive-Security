#!/bin/bash

log_file="./Logs/can_data_logs.log"

commands=()
timestamps=()

# Create directories if they don't exist
createFolders() {
    [ ! -d "./Logs" ] && mkdir ./Logs
    [ ! -d "./Graphs" ] && mkdir ./Graphs
}

clearTmpFiles() {
    rm -rf ./Logs/*.log
    rm -rf ./Graphs/*.png
}

createEmptyDoSTrafficFile() {
    touch ./Logs/dos_traffic.log
}


runGenLogScript() {
    python3 ./log_gen_agent.py
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
    createEmptyDoSTrafficFile
    runGenLogScript
    killCanDump
    postProcessLogs
}

main
