import re
import queue
import time

def calculate_crc(data):
    crc = 0x0000
    poly = 0x4599

    for bit in data:
        crc ^= (int(bit) & 0x01) << 14
        for _ in range(15):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
        crc &= 0x7FFF

    return crc

def stuff_bits(binary_string):
    result = ''
    count = 0

    for bit in binary_string:
        result += bit
        if bit == '0':
            count += 1
            if count == 5:
                result += '1'
                count = 0
        else:
            count = 0

    return result

def hex_to_bits(hex_value, num_bits):
    return bin(int(hex_value, 16))[2:].zfill(num_bits)

def convert_to_binary_string(can_id, dlc, data):
    start_of_frame = '0'
    can_id_bits = hex_to_bits(can_id, 11)
    rtr_bit = '0'
    ide_bit = '0'
    control_r0_bit = '0'
    dlc_bits = bin(dlc)[2:].zfill(4)

    if data[0] != '':
        data_bits = ''.join(hex_to_bits(hex_byte, 8) for hex_byte in data)
    else:
        data_bits = ''

    crc_bit = bin(calculate_crc(start_of_frame + can_id_bits + rtr_bit + ide_bit + control_r0_bit 
                                + dlc_bits + data_bits))[2:].zfill(15)

    crc_delimiter = '1'
    ack_bit = '0'
    ack_delimiter = '1'
    end_of_frame_bits = '1' * 7
    inter_frame_spacing_bits = '1' * 3

    return stuff_bits(start_of_frame + can_id_bits + rtr_bit + ide_bit + control_r0_bit + dlc_bits + data_bits + crc_bit) + crc_delimiter + ack_bit + ack_delimiter + end_of_frame_bits + inter_frame_spacing_bits

def parse_can_log_line(line):
    match = re.match(r'\((\d+\.\d+)\)\s+(\w+)\s+(\w+)\s+\[(\d+)\]\s+([\dA-F\s]+)', line)
    if match:
        timestamp = match.group(1)
        interface = match.group(2)
        can_id = match.group(3)
        dlc = int(match.group(4))
        data = match.group(5).strip().split()
        return {
            'Timestamp': timestamp,
            'Interface': interface,
            'CAN_ID': can_id,
            'DLC': dlc,
            'Data': data
        }
    return None

def read_can_logs(file_path, log_queue, log_type):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        parsed_line = parse_can_log_line(line)
        if parsed_line:
            parsed_line['can_frame'] = convert_to_binary_string(parsed_line['CAN_ID'], parsed_line['DLC'], parsed_line['Data'])
            parsed_line['Type'] = log_type
            log_queue.put(parsed_line)

def arbitration(benign_can_queue, dos_can_queue):
    simulation_time = 0.0
    temp_list = []
    
    with open('./Logs/arbitrated_can_logs.log', 'w') as output_file:
        # Open file to write elapsed times
        with open('arbitration_times.txt', 'w') as elapsed_file:
            while not benign_can_queue.empty() or not dos_can_queue.empty() or temp_list:

                start_time = time.time()
                while not benign_can_queue.empty() and simulation_time >= float(benign_can_queue.queue[0]['Timestamp']):
                    temp_list.append(benign_can_queue.get())

                while not dos_can_queue.empty() and simulation_time >= float(dos_can_queue.queue[0]['Timestamp']):
                    temp_list.append(dos_can_queue.get())

                if not temp_list:
                    if not benign_can_queue.empty() and not dos_can_queue.empty():
                        benign_timestamp = float(benign_can_queue.queue[0]['Timestamp'])
                        dos_timestamp = float(dos_can_queue.queue[0]['Timestamp'])
                        
                        if benign_timestamp < dos_timestamp:
                            simulation_time = benign_timestamp
                            temp_list.append(benign_can_queue.get())
                        elif dos_timestamp < benign_timestamp:
                            simulation_time = dos_timestamp
                            temp_list.append(dos_can_queue.get())
                        else:
                            simulation_time = dos_timestamp
                            temp_list.append(dos_can_queue.get())
                            temp_list.append(benign_can_queue.get())
                    elif not benign_can_queue.empty():
                        simulation_time = float(benign_can_queue.queue[0]['Timestamp'])
                        temp_list.append(benign_can_queue.get())
                    elif not dos_can_queue.empty():
                        simulation_time = float(dos_can_queue.queue[0]['Timestamp'])
                        temp_list.append(dos_can_queue.get())
                
                # Sort temp_list based on CAN_ID and Type ('dos' considered smaller)
                temp_list.sort(key=lambda x: (int(x['CAN_ID'], 16), x['Type'] == 'benign'))

                # Print the simulation time and temp_list
                # print(f"Simulation Time: {simulation_time:.6f}")
                # for packet in temp_list:
                #     print(packet)
                
                # print("")

                if temp_list:
                    smallest_can_packet = temp_list.pop(0)
                    can_data_frame = smallest_can_packet['can_frame']
                    can_data_size = len(can_data_frame)
                    output_file.write(f"({simulation_time:.6f}) {smallest_can_packet['Interface']} {smallest_can_packet['CAN_ID']} [{smallest_can_packet['DLC']}] {' '.join(smallest_can_packet['Data'])}\n")
                    #Bus Speed = 500kb/s 
                    #IFS = 3bits
                    simulation_time += can_data_size / 500000
                    # print("Simulation Time at each step: ", simulation_time)
                    # print("")
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                elapsed_file.write(f"{elapsed_time:.6f}\n")


if __name__ == "__main__":

    benign_can_queue = queue.Queue()
    dos_can_queue = queue.Queue()

    read_can_logs('./Logs/can_logs_jittered.log', benign_can_queue, 'benign')
    read_can_logs('./Logs/dos_traffic.log', dos_can_queue, 'dos')

    arbitration(benign_can_queue, dos_can_queue)
    print("Arbitration completed. Arbitrated output saved to 'arbitrated_can_logs.log'.")
