import os
import time
import torch
import pandas as pd
import explainers.SimpleGradients as grad
from models import BBDNN
from GAME4EXE import DOSHeaderXAIEvasion
from preprocessing.FilePreprocessor import load_and_preprocess_file


seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_adversarials(conf : dict):

    mode = conf["mode"]
    malware_path = conf["malware_path"]
    target_path = conf["target_path"]
    output_folder = conf["output_folder"]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    n_steps = int(conf["n_steps"])
    patience = int(conf["patience"])
    lr = float(conf["lr"])


    model = BBDNN.load_model("./models/BBDNN_model.pth").eval().to(device)
    input_size = 102_400
    explainer = grad.SimpleGradients(model)


    if mode == "M1":
        lambda_p = [1.0]
        lambda_x = [0.0]
    elif mode == "M2":
        lambda_p = [0.0]
        lambda_x = [1.0]
    elif mode == "M3":
        lambda_p = [0.5, 0.55, 0.6, 0.65, 0.7]
        lambda_x = [1 - l for l in lambda_p]
        

    attack = DOSHeaderXAIEvasion(model, softplus_beta=10.0)
    print("----- ORIGINAL TARGET MODEL -----")
    print(model)

    malware_list = os.listdir(malware_path)
    n_files = len(malware_list)
    print("N. files in folder %s: %i" % (malware_path, n_files))

    
    # Verify if they are actual malwares
    malwares_to_manipulate = []
    for f in malware_list:
        m, _ = load_and_preprocess_file(os.path.join(malware_path, f), max_dim=input_size)

        m_conf = model(m.to(device), is_embedded = False).item()
        m_p = 0 if m_conf <= 0.5 else 1
        print("Malware %s prediction: %i, confidence: %f" % (f, m_p, m_conf))
        if m_p == 1:
            malwares_to_manipulate.append(f)
        else:
            print("%s skipped." % f)
    print("N. malwares to manipulate: %i" % len(malwares_to_manipulate))

    mse = torch.nn.MSELoss(reduction="mean")

    gen_report = {
        "filename" : [],
        "lambda_p" : [],
        "lambda_x": [],
        "total_losses" : [],
        "pred_losses" : [],
        "expl_losses" : [],
        "changed_bytes" : [],
        "max_changable_bytes": [],
        "prediction" : [],
        "expl_mse_orig-adv" : [],
        "expl_mse_orig-target" : [],
        "expl_mse_adv-target" : [],
    }

    # Manipulate the malwares
    for i,f in zip(range(len(malwares_to_manipulate)), malwares_to_manipulate):

        print("[%i/%i] Manipulating %s" %(i+1, n_files, f))
        for l_p, l_x in zip(lambda_p, lambda_x):

            print("lambda_p: %f \t lambda_x: %f" % (l_p, l_x))

            report = attack.generate_adversarial(
                malware_path =  os.path.join(malware_path, f),
                goodware_path = target_path,
                output_path = os.path.join(output_folder, f),
                target_label = 0,
                input_size = input_size,
                n_steps = n_steps,
                lr = lr,
                lambda_p = l_p,
                lambda_x = l_x,
                seed = seed,
                patience = patience,
                verbose=True,
            )

            a, _ = load_and_preprocess_file(os.path.join(output_folder, f), max_dim=input_size, pad_value=0)
            a = a.to(device)
            adv_conf = model(a, is_embedded = False).item()
            adv_p = 0 if adv_conf <= 0.5 else 1
            print("Adversarial prediction: %i, confidence: %f" % (adv_p, adv_conf))

            if adv_p == 1:
                print("Adversarial failed to evade, removing.")
                os.remove(os.path.join(output_folder, f))
            else:
                print("Generation successful for lambda_p: %f \t lambda_x: %f" % (l_p, l_x))
                break
            if not report:
                continue
        

        gen_report["filename"].append(report["filename"])
        gen_report["lambda_p"].append(l_p)
        gen_report["lambda_x"].append(l_x)
        gen_report["total_losses"].append(report["total_loss"])
        gen_report["pred_losses"].append(report["pred_loss"])
        gen_report["expl_losses"].append(report["expl_loss"])
        gen_report["changed_bytes"].append(report["changed_bytes"])
        gen_report["max_changable_bytes"].append(report["max_changable_bytes"])

        m, _ = load_and_preprocess_file(os.path.join(malware_path, f), max_dim=input_size, pad_value=0)
        g, _ = load_and_preprocess_file(target_path, max_dim=input_size, pad_value=0)

        m = m.to(device)
        g = g.to(device)
        a = a.to(device)
        

        gen_report["prediction"].append(adv_p)

        m_attr = explainer.attribute(model.embed(m)).flatten()
        g_attr = explainer.attribute(model.embed(g)).flatten()
        a_attr = explainer.attribute(model.embed(a)).flatten()        

        attr_mse_adv_orig = mse(a_attr, m_attr).item()
        attr_mse_orig_good = mse(m_attr, g_attr).item()
        attr_mse_adv_good = mse(a_attr, g_attr).item()

        print("Explanation MSE between ADV and orig. MALWARE: %1.16f" % attr_mse_adv_orig)
        print("Explanation MSE between orig. MALWARE and target GOODWARE: %1.16f" % attr_mse_orig_good)
        print("Explanation MSE between ADV and target GOODWARE: %1.16f\n" % attr_mse_adv_good)

        gen_report["expl_mse_orig-adv"].append(attr_mse_adv_orig)
        gen_report["expl_mse_orig-target"].append(attr_mse_orig_good)
        gen_report["expl_mse_adv-target"].append(attr_mse_adv_good)

        torch.cuda.empty_cache()


    # Report creation
    if not os.path.exists("./reports/"):
        os.mkdir("./reports/")
    df = pd.DataFrame(gen_report)
    report_filename = f"./reports/generation_{mode}_{int(time.time())}.csv"
    df.to_csv(report_filename, index=False)
    print("Done.")



