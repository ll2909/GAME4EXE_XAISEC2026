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

    model = OriginalMalConv.load_model("./models/original_malconv_model.pth").eval().to(device)
    input_size = 2**20

    report = {
        "filename" : [],
        "prediction" : []
        }


    for adv in os.listdir(advs_path):
        
        adv_tens, _ = load_and_preprocess_file(os.path.join(advs_path, adv), max_dim=input_size)
        adv_tens = model.embed(adv_tens.to(device))
        adv_conf = model(adv_tens, is_embedded = True).item()
        adv_pred = 0 if adv_conf < 0.5 else 1
        print("Adversarial predicion: %i  -  Condifence:  %f" % (adv_pred, adv_conf))

        report["filename"].append(adv)
        report["prediction"].append(adv_pred)

        torch.cuda.empty_cache()


    # Report creation
    if not os.path.exists("./reports/"):
        os.mkdir("./reports/")
    df = pd.DataFrame(report)
    report_filename = f"./reports/transferibility_bbdnn_to_malconv_{int(time.time())}.csv"
    df.to_csv(report_filename, index=False)
    print("Done.")

    

