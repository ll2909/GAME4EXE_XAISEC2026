import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from configparser import ConfigParser


if __name__ == "__main__":

    cfg_parser = ConfigParser()
    cfg_parser.read("config.conf")
    config = dict(cfg_parser["BOXPLOT_SINGLE"])
    report_path = config["report_path"]
    
    report = pd.read_csv(report_path)

    n_samples = len(report)

    success = report[report["prediction"] == 0]
    print("Success rate: %f - [%i/%i]" % (len(success)/n_samples, len(success), n_samples))

    effective_m1 = success[success["expl_mse_adv-target"] < success["expl_mse_orig-adv"]]
    print("Effectively manipulated samples: %f - [%i/%i]" % (len(effective_m1)/len(success), len(effective_m1), len(success)))

    max_value = max(max(success["expl_mse_orig-target"]), max(success["expl_mse_adv-target"]))

    # Create the figure
    fig = go.Figure()

    # Add Apple boxes
    fig.add_trace(go.Box(
        y=success["expl_mse_orig-target"],
        name='Original - Target',
        marker_color="#603AA7",
        legendgroup='Original - Target',
        showlegend=True,
        offsetgroup='Original - Target'
    ))


    fig.add_trace(go.Box(
        y=success["expl_mse_adv-target"],
        name='Adversarial - Target',
        marker_color="#943636",
        legendgroup='Adversarial - Target',
        showlegend=True,
        offsetgroup='Adversarial - Target'
    ))



    fig.update_layout(
        yaxis_title='XaiMSE',
        boxmode='group',
        template='seaborn',
        plot_bgcolor='#E8E8F0',
        paper_bgcolor='#E8E8F0',
        font=dict(size=28),
        width = 900, height = 650,
        showlegend = False,
        yaxis=dict(
            tickmode = "linear",
            tick0 = 0.0,
            dtick = np.round((max_value / 5), decimals=6),
            exponentformat='power',
            showexponent='all',
        ),
    )


    fig.show()
    #fig.write_image('./bbdnn_boxplots_grid.png', format = 'png')
