"""
Run the full 4-round spoof pipeline for each IDS sequentially.
Each IDS gets its own output dirs so rounds are independent.
Usage: python run_all_ids_spoof.py [--config config_spoof_CH.yaml] [--rounds 4]
"""
import os
import sys
import shutil
import yaml
import argparse
from io import StringIO

# ---------------------------------------------------------------
# IDS registry — add / comment-out entries as needed
# ---------------------------------------------------------------
IDS_REGISTRY = [
    {
        "name":       "densenet161",
        "module":     "scripts.evaluate_spoof_CH_densenet161",
        "model_path": "Trained_models/Densenet161_car_hacking_spoof_withdata.pt",
    },
    {
        "name":       "resnet50",
        "module":     "scripts.evaluate_spoof_CH_resnet50",
        "model_path": "Trained_models/ResNet50_car_hacking_spoof.pt",
    },
    {
        "name":       "ri_resnet",
        "module":     "scripts.evaluate_spoof_CH_resnet",
        "model_path": "Trained_models/spoof_final_model.h5",
    },
    {
        "name":       "entropy",
        "module":     "scripts.evaluate_spoof_Entropy",
        "model_path": None,
    },
    {
        "name":       "seq",
        "module":     "scripts.evaluate_spoof_seq_based",
        "model_path": "Trained_models/car_hacking_transitions_1.pkl",
    },
    {
        "name":       "mulsam",
        "module":     "scripts.evaluate_gear_CH_MULSAM",
        "model_path": "./../MULSAM_CH/outputs/mulsam_spoof_target.pth",
    },
]


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_one_round(config, ids, round_num):
    """Run attack → decode → evaluate → update for one IDS, one round."""

    name = ids["name"]
    steps = config.get("run_steps", {})

    # Per-IDS output dirs (appended with IDS name)
    atk_out   = f'{config["attack"]["output_dir"]}_{name}'
    dec_out   = f'{config["decode"]["decoded_output_dir"]}_{name}'
    pred_out  = f'{config["evaluate"]["prediction_output_dir"]}_{name}'
    track_dir = f'{config["update"]["tracksheet_dir"]}_{name}'

    print(f"\n  --- Round {round_num} ---")

    # ======================================================
    # STEP 1: ATTACK
    # ======================================================
    if steps.get("attack", False):
        import importlib
        run_attack = importlib.import_module("scripts.adversarial_attack_spoof_CH").run

        attack_cfg = config["attack"].copy()
        attack_cfg["output_path"] = atk_out
        attack_cfg["model_path"]  = config["attack"]["surrogate_model"]
        attack_cfg["rounds"]      = round_num

        if round_num == 0:
            attack_cfg["test_data_dir"]     = config["attack"]["original_test_dir"]
            attack_cfg["packet_level_data"] = config["attack"]["original_tracksheet"]
            attack_cfg["test_label_file"]   = config["attack"]["original_label_file"]
        else:
            attack_cfg["test_data_dir"]     = atk_out
            attack_cfg["packet_level_data"] = f"{track_dir}/spoof_test_track_{round_num-1}.csv"
            attack_cfg["test_label_file"]   = f"{atk_out}/labels_{round_num}.txt"

        os.makedirs(atk_out, exist_ok=True)

        stats_file = os.path.join(atk_out, f"stats_round_{round_num}.txt")
        old_stdout = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            run_attack(attack_cfg)
        finally:
            sys.stdout = old_stdout
        with open(stats_file, "w") as f:
            f.write(buf.getvalue())
        print(f"    [attack] done → {atk_out}")

    # ======================================================
    # STEP 2: DECODE
    # ======================================================
    if steps.get("decode", False):
        import importlib
        run_decode = importlib.import_module("scripts.Traffic_decoder_spoof_CH").run

        decode_cfg = config["decode"].copy()
        decode_cfg["rounds"]       = round_num
        decode_cfg["input_images"] = atk_out
        decode_cfg["csv_file"]     = f"{atk_out}/packet_level_data_{round_num}.csv"
        decode_cfg["output_file"]  = f"{dec_out}/traffic_{round_num}.txt"

        os.makedirs(dec_out, exist_ok=True)
        run_decode(decode_cfg)
        print(f"    [decode] done → {dec_out}/traffic_{round_num}.txt")

    # ======================================================
    # GENERATE perturbed_labels.txt  (image_N → perturbed_image_N)
    # ======================================================
    if steps.get("attack", False) and round_num == 0 and name in ("densenet161", "resnet50"):
        dst_label_file = f"{atk_out}/perturbed_labels.txt"
        with open(config["attack"]["original_label_file"], "r") as src, \
             open(dst_label_file, "w") as dst:
            for line in src:
                dst.write(line.replace("image_", "perturbed_image_", 1))

    # ======================================================
    # STEP 3: EVALUATE
    # ======================================================
    if steps.get("evaluate", False):
        import importlib
        run_eval = importlib.import_module(ids["module"]).run

        os.makedirs(pred_out, exist_ok=True)
        os.makedirs(track_dir, exist_ok=True)

        eval_cfg = {
            "rounds":         round_num,
            "traffic_path":   f"{dec_out}/traffic_{round_num}.txt",
            "tracksheet":     f"{atk_out}/packet_level_data_{round_num}.csv",
            "output_path":    f"{pred_out}/prediction_output_{round_num}.csv",
            "tracksheet_dir": track_dir,
            "attack_output_dir": atk_out,
        }
        if ids["model_path"] is not None:
            eval_cfg["model_path"] = ids["model_path"]

        if name in ("densenet161", "resnet50"):
            eval_cfg["test_dataset_dir"] = atk_out
            eval_cfg["test_label_file"]  = f"{atk_out}/perturbed_labels.txt"

        run_eval(eval_cfg)
        print(f"    [evaluate:{name}] done → {pred_out}/prediction_output_{round_num}.csv")

    # ======================================================
    # STEP 4: UPDATE
    # ======================================================
    if steps.get("update", False):
        import importlib
        run_update = importlib.import_module("scripts.update_labels_spoof_CH").run

        os.makedirs(track_dir, exist_ok=True)

        update_cfg = config["update"].copy()
        update_cfg["tracksheet"]         = f"{track_dir}/spoof_test_track_{round_num}.csv"
        update_cfg["label_file"]         = config["attack"]["original_label_file"]
        update_cfg["updated_label_file"] = f"{atk_out}/labels_{round_num+1}.txt"

        run_update(update_cfg)
        print(f"    [update] done → {track_dir}/spoof_test_track_{round_num}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_spoof_CH.yaml")
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)

    for ids in IDS_REGISTRY:
        print(f"\n{'='*50}")
        print(f"  IDS: {ids['name']}")
        print(f"{'='*50}")
        for round_num in range(args.rounds):
            run_one_round(config, ids, round_num)
