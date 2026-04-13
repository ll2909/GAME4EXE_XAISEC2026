import os
import torch
import numpy as np
from configparser import ConfigParser
import explainers.SimpleGradients as grad
from models import BBDNN, OriginalMalConv
from preprocessing.FilePreprocessor import load_and_preprocess_file
from utils.RNG import set_reproducibility
import sys
from tqdm import tqdm

from plotly.subplots import make_subplots
import plotly.graph_objects as go

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    path = sys.argv[1]
    print("Config path: ", path)
    
    cfg_parser = ConfigParser()
    cfg_parser.read(path)
    config = dict(cfg_parser["CONFIG"])

    set_reproducibility(42)

    target_path = config["target_path"]
    malware_path = config["malware_path"]
    advs_path = config["output_folder"]

    print("Using model: ", config["model"])
    
    if config["model"] == "bbdnn":
        model = BBDNN.load_model("./models/BBDNN_model.pth").eval().to(device)
        input_size = 102_400
    else:
        model = OriginalMalConv.load_model("./models/original_malconv_model.pth").eval().to(device)
        input_size = 2**20

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


    for a in tqdm(os.listdir(advs_path)):
        adv_tensor, _ = load_and_preprocess_file(os.path.join(advs_path, a), input_size)
        malware_tensor, _ = load_and_preprocess_file(os.path.join(malware_path, a), input_size)
        

        malw_emb = model.embed(malware_tensor.to(device))
        malw_expl = explainer.attribute(malw_emb).squeeze().detach()


        adv_emb = model.embed(adv_tensor.to(device))
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
                      margin=dict(l=0, r=0, t=0, b=0),
                      plot_bgcolor='#E8E8F0',
                      paper_bgcolor='#E8E8F0',
                      height=600, width=1000,
                      font=dict(size=30),
                      xaxis=dict(title=dict(text="Byte Position")), 
                      yaxis=dict(title=dict(text="ByteXaiMSE")),
                      yaxis_range=[0, 0.0016],
                      showlegend = False
                      )
    
    fig.update_layout(
    xaxis2=dict(
        domain=[0.4, 0.95],
        anchor='y2',
        range=[0, 1008],
    ),
    yaxis2=dict(
        domain=[0.5, 0.95],
        anchor='x2',
        range=[0, 0.0016],
    ))

    fig.add_trace(go.Scatter(x = [i for i in range(input_size)], y=global_mse_mt, xaxis='x2', yaxis='y2', line = dict(color = "#603AA7", width = 2)))
    fig.add_trace(go.Scatter(x = [i for i in range(input_size)], y=global_mse_at, xaxis='x2', yaxis='y2', line = dict(color = "#943636", width = 2)))
    fig.add_shape(
        type="line",
        xref="x2", 
        yref="y2",
        x0=234.24, x1=234.24,
        y0=0, y1=1,
        line=dict(color="black", width=2, dash="dash")
    )
    
    #fig.show()
    fig.write_image(f"./{config['model']}_bytexai_mse.png", format="png")