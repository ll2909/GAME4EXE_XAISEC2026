import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from configparser import ConfigParser


if __name__ == "__main__":

    cfg_parser = ConfigParser()
    cfg_parser.read("config.conf")
    config = dict(cfg_parser["BOXPLOT_MULTIPLE"])
    report_path_m1 = config["m1_report_path"]
    report_path_m2 = config["m2_report_path"]
    report_path_m3 = config["m3_report_path"]

    report_m1 = pd.read_csv(report_path_m1)
    report_m2 = pd.read_csv(report_path_m2)
    report_m3 = pd.read_csv(report_path_m3)


    titles = ["M1", "M2", "M3"]


    n_samples = len(report_m1)
    success_m1 = report_m1[report_m1["prediction"] == 0]
    success_m2 = report_m2[report_m2["prediction"] == 0]
    success_m3 = report_m3[report_m3["prediction"] == 0]
    print("Success rate m1: %f - [%i/%i]" % (len(success_m1)/n_samples, len(success_m1), n_samples))
    print("Success rate m2: %f - [%i/%i]" % (len(success_m2)/n_samples, len(success_m2), n_samples))
    print("Success rate m3: %f - [%i/%i]" % (len(success_m3)/n_samples, len(success_m3), n_samples))


    effective_m1 = success_m1[success_m1["expl_mse_adv-target"] < success_m1["expl_mse_orig-adv"]]
    effective_m2 = success_m2[success_m2["expl_mse_adv-target"] < success_m2["expl_mse_orig-adv"]]
    effective_m3 = success_m3[success_m3["expl_mse_adv-target"] < success_m3["expl_mse_orig-adv"]]
    print("Effectively manipulated samples m1: %f - [%i/%i]" % (len(effective_m1)/len(success_m1), len(effective_m1), len(success_m1)))
    print("Effectively manipulated samples m2: %f - [%i/%i]" % (len(effective_m2)/len(success_m2), len(effective_m2), len(success_m2)))
    print("Effectively manipulated samples m3: %f - [%i/%i]" % (len(effective_m3)/len(success_m3), len(effective_m3), len(success_m3)))

    max_value = max(max(success_m1["expl_mse_orig-target"]), max(success_m2["expl_mse_orig-target"]), max(success_m3["expl_mse_orig-target"]),
                    max(success_m1["expl_mse_adv-target"]), max(success_m2["expl_mse_adv-target"]), max(success_m3["expl_mse_adv-target"]))

    # Create the figure
    fig = go.Figure()

    fig.add_trace(go.Box(
        y=success_m1["expl_mse_orig-target"],
        name='Original - Target',
        x=[titles[0]] * len(success_m1["expl_mse_orig-target"]),
        marker_color="#603AA7",
        legendgroup='Original - Target',
        showlegend=True,
        offsetgroup='Original - Target'
    ))

    fig.add_trace(go.Box(
        y=success_m2["expl_mse_orig-target"],
        name='Original - Target',
        x=[titles[1]] * len(success_m2["expl_mse_orig-target"]),
        marker_color='#603AA7',
        legendgroup='Original - Target',
        showlegend=False,
        offsetgroup='Original - Target'
    ))

    fig.add_trace(go.Box(
        y=success_m3["expl_mse_orig-target"],
        name='Original - Target',
        x=[titles[2]] * len(success_m3["expl_mse_orig-target"]),
        marker_color='#603AA7',
        legendgroup='Original - Target',
        showlegend=False,
        offsetgroup='Original - Target'
    ))


    fig.add_trace(go.Box(
        y=success_m1["expl_mse_adv-target"],
        name='Adversarial - Target',
        x=[titles[0]] * len(success_m1["expl_mse_adv-target"]),
        marker_color="#943636",
        legendgroup='Adversarial - Target',
        showlegend=True,
        offsetgroup='Adversarial - Target'
    ))

    fig.add_trace(go.Box(
        y=success_m2["expl_mse_adv-target"],
        name='Adversarial - Target',
        x=[titles[1]] * len(success_m2["expl_mse_adv-target"]),
        marker_color='#943636',
        legendgroup='Adversarial - Target',
        showlegend=False,
        offsetgroup='Adversarial - Target'
    ))

    fig.add_trace(go.Box(
        y=success_m3["expl_mse_adv-target"],
        name='Adversarial - Target',
        x=[titles[2]] * len(success_m3["expl_mse_adv-target"]),
        marker_color='#943636',
        legendgroup='Adversarial - Target',
        showlegend=False,
        offsetgroup='Adversarial - Target'
    ))
    

    # Update layout
    fig.update_layout(
        xaxis_title='Configurations',
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
        )
    )

    fig.show()
    #fig.write_image('./boxplots_multiple_conf.png', format = 'png')
