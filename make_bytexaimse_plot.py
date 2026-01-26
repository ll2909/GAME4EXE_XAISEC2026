import os
import torch
import numpy as np
from configparser import ConfigParser
import explainers.SimpleGradients as grad
from models import BBDNN
from preprocessing.FilePreprocessor import load_and_preprocess_file
from utils.RNG import set_reproducibility

from plotly.subplots import make_subplots
import plotly.graph_objects as go

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    
    cfg_parser = ConfigParser()
    cfg_parser.read("config.conf")
    config = dict(cfg_parser["CONFIG"])

    set_reproducibility(42)

    target_path = config["target_path"]
    malware_path = config["malware_path"]
    advs_path = config["output_path"]

    model = BBDNN.load_model("./models/BBDNN_model.pth").eval().to(device)
    input_size = 102_400
    explainer = grad.SimpleGradients(model)


    print("Target goodware: %s" % target_path)
    target_tensor, _ = load_and_preprocess_file(target_path, input_size)
    target_emb = model.embed(target_tensor.to(device))
    target_conf = model(target_emb, is_embedded = True).item()
    target_pred = 0 if target_conf < 0.5 else 1
    print("Target goodware prediction: %i \t confidence: %f" % (target_pred, target_conf))
    target_expl = explainer.attribute(target_emb).squeeze()

    global_mse_mt = np.array([0.0 for _ in range(input_size)])
    global_mse_at = np.array([0.0 for _ in range(input_size)])


    for a in os.listdir(advs_path):
        print("For adversarial/malware file: %s" % a)

        adv_tensor, _ = load_and_preprocess_file(os.path.join(advs_path, a), input_size)
        malware_tensor, _ = load_and_preprocess_file(os.path.join(malware_path, a), input_size)
        

        malw_emb = model.embed(malware_tensor.to(device))
        malw_conf = model(malw_emb, is_embedded = True).item()
        malw_pred = 0 if malw_conf < 0.5 else 1
        print("Original malware prediction: %i \t confidence: %f" % (malw_pred, malw_conf))
        malw_expl = explainer.attribute(malw_emb).squeeze().detach()


        adv_emb = model.embed(adv_tensor.to(device))
        adv_conf = model(adv_emb, is_embedded = True).item()
        adv_pred = 0 if adv_conf < 0.5 else 1
        print("Adversarial malware prediction: %i \t confidence: %f" % (adv_pred, adv_conf))
        adv_expl = explainer.attribute(adv_emb).squeeze().detach()

        # We compute the byte-wise mse for each input byte
        malware_target_expl_distance = torch.nn.functional.mse_loss(malw_expl, target_expl, reduction="none").mean(dim=1).cpu().detach().numpy()
        adv_target_expl_distance = torch.nn.functional.mse_loss(adv_expl, target_expl, reduction="none").mean(dim=1).cpu().detach().numpy()

        # we accumulate the explanation distances, for producing a global explanation difference
        global_mse_mt = global_mse_mt + malware_target_expl_distance
        global_mse_at = global_mse_at + adv_target_expl_distance

        torch.cuda.empty_cache()

    
    # For the global bytexaimse we average the accumulated mse for the number of adversarials
    n_advs = len(os.listdir(advs_path))
    global_mse_mt = global_mse_mt / n_advs
    global_mse_at = global_mse_at / n_advs


    fig = make_subplots(rows = 1, cols = 1)
    fig.add_trace(
        go.Scatter(x = [i for i in range(input_size)], y=global_mse_mt, name="Global Original - Target", mode = "lines", line = dict(color = "#603AA7", width = 2)), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x = [i for i in range(input_size)], y=global_mse_at, name="Global Adversarial - Target", mode = "lines", line = dict(color = "#943636", width = 2)), row=1, col=1
    )
    fig.update_layout(template='seaborn',
                      plot_bgcolor='#E8E8F0',
                      paper_bgcolor='#E8E8F0',
                      height=600, width=1000,
                      font=dict(size=30),
                      xaxis=dict(title=dict(text="Byte Position")), 
                      yaxis=dict(title=dict(text="ByteXaiMSE")),
                      showlegend = True
                      )
    
    
    fig.show()
    #fig.write_image("./global_bytexai_mse.png", format="png")