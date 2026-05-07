import pandas as pd
import os 
import yaml

# def update_track(packet_level_data, prediction_file, updated_track_file):

#     with open(prediction_file, 'r') as prediction_f, open(packet_level_data, 'r') as packet_f, open(updated_track_file, 'w') as output_f :
#         next(prediction_f)  # Skip header line
#         next(packet_f)       # Skip header line
#         #write header to output file
#         output_f.write('row_no,timestamp,can_id,image_no,valid_flag,label' + '\n')
#         for pred_line, packet_line in zip(prediction_f, packet_f):
#             pred_parts = pred_line.strip().split(',')
#             packet_parts = packet_line.strip().split(',')
#             # print("Pred parts:", pred_parts)
#             # print("Packet parts:", packet_parts)

#             # if(int(pred_parts[1],16) == int(packet_parts[2],16)):
#             packet_parts = packet_parts[:-2] + ["1" if pred_parts[-1] == 'A' else "0"] 
#             updated_packet_line = ','.join(packet_parts)
#             # print(updated_packet_line)
#             output_f.write(updated_packet_line + '\n')
#         # lines in packet_f > lines in prediction_f, so no need to handle extra lines in packet_f
#         while(True):
#             line = packet_f.readline()
#             if not line:
#                 break
#             part = line.strip().split(',')
#             output_f.write(','.join(part[:-1]))
#             output_f.write('\n')  # default label 0 for packets with no prediction
            
    
def update_labels(updated_track_file, label_file, updated_label_file):
    
    df = pd.read_csv(updated_track_file)
    df["row_no"] = df["row_no"].astype(int)
    df["timestamp"] = df["timestamp"].astype(float)
    df["image_no"] = df["image_no"].astype(int)
    df["valid_flag"] = df["valid_flag"].astype(int)
    # print("DF rows:", len(df))
    # print("Unique images:", df['image_no'].nunique())
    with open(label_file, 'r') as label_f, open(updated_label_file, 'w') as final_label_f:
        # next(updated_f)  # Skip header line
        # group by image_no in packet level data 

        label_line = next(label_f).strip()
        for image_no, group in df.groupby('image_no'):
            
            # labels = group['label'].tolist()
            # if "A" in labels:
            #     final_label_f.write(f"{image_no},1\n")
            # else:
            #     final_label_f.write(f"{image_no},0\n")

            img, rest = label_line.split(":")
            valid_flag = int(rest.split(",")[0])
            
            packet_labels = group['pred_label'].astype(str).str.upper().tolist()
            new_label = 1 if "A" in packet_labels else 0

            
            final_label_f.write(f"perturbed_image_{image_no}.png: {valid_flag}, {new_label}\n")
            try:
                label_line = next(label_f).strip()
            except StopIteration:
                break


        

def run(params):

    tracksheet = params["tracksheet"]
    label_file = params["label_file"]
    updated_label_file = params["updated_label_file"]

    update_labels(tracksheet,label_file,updated_label_file)
    print("updated label file")


if __name__ == "__main__":

    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_spoof_CH.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # Ensure attack section exists
    if "update" not in cfg:
        raise ValueError("Config file must contain 'update' section.")

    run(cfg["update"])