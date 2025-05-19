#!/bin/bash

gen_log_file="./Logs/can_data_logs.log"
rep_log_file="./Logs/can_data_replay.log"

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

runGenLogScript() {
    python3 ./log_gen_agent.py
}

runReplayScript() {
    python3 ./log_replay_agent.py
}

startCanDump() {
    candump -tz vcan0 > "$gen_log_file" &
    sleep 1s
}

startCanDumpReplay() {
    candump -tz vcan0 > "$rep_log_file" &
    sleep 1s
}

killCanDump() {
    pkill -9 candump
}

postProcessLogs() {
    sed -i 's/^ //' "$gen_log_file"
    # python3 arbitration.py
    # python3 merge_can_logs.py
    # python3 ./update_can_logs.py
}

postProcessLogsReplay() {
    sed -i 's/^ //' "$rep_log_file"
}

main() {
    createFolders
    clearTmpFiles

    # Generate log data
    # startCanDump
    runGenLogScript
    # killCanDump
    # postProcessLogs

    # Generate replay data
    # startCanDumpReplay
    runReplayScript
    # killCanDump
    # postProcessLogsReplay
}

main
