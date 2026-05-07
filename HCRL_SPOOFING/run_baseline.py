import sys, os
sys.path.insert(0, os.path.dirname(__file__))

TRACKSHEET = "tracksheets_CH/gear_test_track_original.csv"
OUTPUT_DIR  = "prediction_output_spoof_CH"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- RI_ResNet ---
try:
    import scripts.evaluate_spoof_CH_resnet as _m_resnet
    _m_resnet.save_preds = lambda *a, **kw: None
    _m_resnet.run({"rounds": -1,
                   "model_path": "Trained_models/spoof_final_model.h5",
                   "traffic_path": "CAN_DATA/gear_test.csv",
                   "tracksheet":   TRACKSHEET,
                   "output_path":  f"{OUTPUT_DIR}/baseline_resnet.csv"})
except Exception as e:
    print(f"[run_baseline] RI_ResNet FAILED: {e}")

# --- ResNet50 ---
try:
    import scripts.evaluate_spoof_CH_resnet50 as _m_r50
    _m_r50.save_preds = lambda *a, **kw: None
    _m_r50.run({"rounds": -1,
                "model_path":      "Trained_models/ResNet50_car_hacking_spoof.pt",
                "test_dataset_dir":"gear_test_images",
                "test_label_file": "gear_test_images/labels.txt",
                "tracksheet":       TRACKSHEET,
                "output_path":      f"{OUTPUT_DIR}/baseline_resnet50.csv"})
except Exception as e:
    print(f"[run_baseline] ResNet50 FAILED: {e}")

# --- DenseNet161 ---
try:
    import scripts.evaluate_spoof_CH_densenet161 as _m_d161
    _m_d161.save_preds = lambda *a, **kw: None
    _m_d161.run({"rounds": -1,
                 "model_path":      "Trained_models/Densenet161_car_hacking_spoof_withdata.pt",
                 "test_dataset_dir":"gear_test_images",
                 "test_label_file": "gear_test_images/labels.txt",
                 "tracksheet":       TRACKSHEET,
                 "output_path":      f"{OUTPUT_DIR}/baseline_d161.csv"})
except Exception as e:
    print(f"[run_baseline] DenseNet161 FAILED: {e}")

# --- MULSAM ---
try:
    import scripts.evaluate_gear_CH_MULSAM as _m_mulsam
    _m_mulsam.save_preds = lambda *a, **kw: None
    _m_mulsam.run({"rounds": -1,
                   "model_path":   "Trained_models/mulsam_spoof_target.pth",  # fill in
                   "traffic_path": "CAN_DATA/gear_test.csv",
                   "tracksheet":   TRACKSHEET,
                   "output_path":  f"{OUTPUT_DIR}/baseline_mulsam.csv"})
except Exception as e:
    print(f"[run_baseline] MULSAM FAILED: {e}")

# --- Entropy_IDS ---
try:
    import scripts.evaluate_spoof_Entropy as _m_entropy
    _m_entropy.save_preds = lambda *a, **kw: None
    _m_entropy.run({"rounds": -1,
                    "traffic_path": "CAN_DATA/gear_test.csv",
                    "tracksheet":   TRACKSHEET,
                    "output_path":  f"{OUTPUT_DIR}/baseline_entropy.csv"})
except Exception as e:
    print(f"[run_baseline] Entropy_IDS FAILED: {e}")

# --- Seq_IDS ---
try:
    import scripts.evaluate_spoof_seq_based as _m_seq
    _m_seq.save_preds = lambda *a, **kw: None
    _m_seq.save_predictions_to_txt_file = lambda *a, **kw: None
    _m_seq.run({"rounds": -1,
                "model_path":   "Trained_models/car_hacking_transitions_1.pkl",
                "traffic_path": "CAN_DATA/gear_test.csv",
                "tracksheet":   TRACKSHEET,
                "output_path":  f"{OUTPUT_DIR}/baseline_seq.csv"})
except Exception as e:
    print(f"[run_baseline] Seq_IDS FAILED: {e}")
