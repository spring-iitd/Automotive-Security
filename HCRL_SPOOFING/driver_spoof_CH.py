import os

import sys
import yaml
import argparse
from io import StringIO


# -------------------------------------------------
# Load YAML Config
# -------------------------------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# Main Pipeline
# -------------------------------------------------
def pipeline_run(config):
    round_num = config["round"]
    steps = config.get("run_steps", {})

    print(f"\n==============================")
    print(f"Running Round in driver file {round_num}")
    print(f"==============================")

    # ==========================================================
    # STEP 1: ATTACK
    # ==========================================================
    if steps.get("attack", False):

        from scripts.adversarial_attack_spoof_CH import run as run_attack

        round_num = config["round"]

        # Copy base attack config
        attack_cfg = config["attack"].copy()

        # Required keys for attack script
        attack_cfg["output_path"] = config["attack"]["output_dir"]
        attack_cfg["model_path"] = config["attack"]["surrogate_model"]
        attack_cfg["rounds"] = round_num

        # Round-dependent logic
        if round_num == 0:
            attack_cfg["test_data_dir"] = config["attack"]["original_test_dir"]
            attack_cfg["packet_level_data"] = config["attack"]["original_tracksheet"]
            attack_cfg["test_label_file"] = config["attack"]["original_label_file"]
        else:
            attack_cfg["test_data_dir"] = config["attack"]["output_dir"]
            attack_cfg["packet_level_data"] = (
                f'{config["update"]["tracksheet_dir"]}/spoof_test_track_{round_num-1}.csv'
            )
            attack_cfg["test_label_file"] = (
                f'{config["attack"]["output_dir"]}/labels_{round_num}.txt'
            )

        # ---------------------------------------------------
        # Create output directory
        # ---------------------------------------------------
        attack_out = attack_cfg["output_path"]
        os.makedirs(attack_out, exist_ok=True)

        print("\n=== Step 1: Adversarial Attack ===")

        # ---------------------------------------------------
        # Capture stdout and save to stats file
        # ---------------------------------------------------
        stats_file = os.path.join(attack_out, f"stats_round_{round_num}.txt")

        old_stdout = sys.stdout
        sys.stdout = mystream = StringIO()

        try:
            run_attack(attack_cfg)
        finally:
            sys.stdout = old_stdout

        with open(stats_file, "w") as f:
            f.write(mystream.getvalue())

        print(f"[INFO] Attack log saved to {stats_file}")



    # ==========================================================
    # STEP 2: DECODE
    # ==========================================================
    if steps.get("decode", False):

        from scripts.Traffic_decoder_spoof_CH import run as run_decode

        decode_cfg = config["decode"].copy()
        decode_cfg["rounds"] = round_num
        decode_cfg["input_images"] = config["attack"]["output_dir"]
        decode_cfg["csv_file"] = (
            f'{config["attack"]["output_dir"]}/packet_level_data_{round_num}.csv'
        )
        decode_cfg["output_file"] = (
            f'{config["decode"]["decoded_output_dir"]}/traffic_{round_num}.txt'
        )

        os.makedirs(config["decode"]["decoded_output_dir"], exist_ok=True)

        print("\n=== Step 2: Traffic Decoder ===")
        run_decode(decode_cfg)


    # ==========================================================
    # STEP 3: EVALUATION
    # ==========================================================
    if steps.get("evaluate", False):

        from scripts.evaluate_spoof_CH_densenet161 import run as run_eval

        eval_cfg = config["evaluate"].copy()
        eval_cfg["rounds"] = round_num
        eval_cfg["model_path"] = config["evaluate"]["model_path"]
        eval_cfg["traffic_path"] = (
            f'{config["decode"]["decoded_output_dir"]}/traffic_{round_num}.txt'
        )
        eval_cfg["tracksheet"] = (
            f'{config["attack"]["output_dir"]}/packet_level_data_{round_num}.csv'
        )
        eval_cfg["output_path"] = (
            f'{config["evaluate"]["prediction_output_dir"]}/prediction_output_{round_num}.csv'
        )

        os.makedirs(config["evaluate"]["prediction_output_dir"], exist_ok=True)

        print("\n=== Step 3: Evaluation ===")
        run_eval(eval_cfg)


    # ==========================================================
    # STEP 4: UPDATE
    # ==========================================================
    if steps.get("update", False):

        from scripts.update_labels_spoof_CH import run as run_update

        update_cfg = config["update"].copy()

        update_cfg["tracksheet"] = (
            f'{config["update"]["tracksheet_dir"]}/spoof_test_track_{round_num}.csv'
        )

        # Label logic
        update_cfg["label_file"] = config["attack"]["original_label_file"]

        update_cfg["updated_label_file"] = (
            f'{config["attack"]["output_dir"]}/labels_{round_num+1}.txt'
        )

        print("\n=== Step 4: Update Labels ===")
        run_update(update_cfg)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--config", type=str, default="config_spoof_CH.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["round"] = args.round

    pipeline_run(cfg)