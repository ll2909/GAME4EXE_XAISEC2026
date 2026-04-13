import torch
import pandas as pd
import os
import time
import explainers.SimpleGradients as grad
from models import OriginalMalConv, BBDNN
from utils.RNG import set_reproducibility
from preprocessing.FilePreprocessor import load_and_preprocess_file


set_reproducibility(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


def transferability_test(conf : dict):
    advs_path = conf["output_folder"]
    samples = os.listdir(conf["malware_path"])
    model_selection = conf["model"]
    mode = conf["mode"]

    if model_selection == "bbdnn":
        print("Transferibility test against MalConv")
        model = OriginalMalConv.load_model("./models/original_malconv_model.pth").eval().to(device)
        input_size = 2**20
        target_model = "malconv"
    else:
        print("Transferibility test against BBDNN")
        model = BBDNN.load_model("./models/BBDNN_model.pth").eval().to(device)
        input_size = 102_400
        target_model = "bbdnn"

    report = {
        "filename" : samples,
        "prediction" : [-1 for _ in range(len(samples))]
        }
    report = pd.DataFrame(report)


    for adv in os.listdir(advs_path):
        
        adv_tens, _ = load_and_preprocess_file(os.path.join(advs_path, adv), max_dim=input_size)
        adv_tens = model.embed(adv_tens.to(device))
        adv_conf = model(adv_tens, is_embedded = True).item()
        adv_pred = 0 if adv_conf < 0.5 else 1
        print("Adversarial predicion: %i  -  Condifence:  %f" % (adv_pred, adv_conf))

        report.loc[(report["filename"] == adv), "prediction"] = adv_pred

        torch.cuda.empty_cache()


    # Report creation
    if not os.path.exists("./reports/"):
        os.mkdir("./reports/")
    report_filename = f"./reports/transferibility_{model_selection}_to_{target_model}_{mode}_{int(time.time())}.csv"
    report.to_csv(report_filename, index=False)
    print("Done.")

    

