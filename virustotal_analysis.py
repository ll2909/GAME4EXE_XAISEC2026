import vt
import os
import pandas as pd
import time
import hashlib
from tqdm import tqdm
import pickle
from configparser import ConfigParser


def scan_file(file_path, api_key):
    # scan a file
    report = {
        "n_engines" : 0,
        "malicious" : 0,
        "suspicious" : 0
    }

    
    retry = True

    while retry:
        try:
            with open(file_path, "rb") as f:
                client = vt.Client(api_key)
                analysis = client.scan_file(f, wait_for_completion=True)
                det_stats = analysis.get("stats")
                tot_det = sum(det_stats[k] for k in det_stats.keys())
                score = (det_stats["malicious"] + det_stats["suspicious"]) / tot_det
                time.sleep(60)
                retry = False
                client.close()
        except vt.error.APIError as e:
            print(e)
            print("Error in uploaded file analysis, waiting 5 mins and retrying...")
            client.close()
            time.sleep(5* 60)
            retry = True
        except OSError as e:
            print(e)
            print("An exception has occured, waiting 1 min and retrying...")
            client.close()
            time.sleep(1* 60)
            retry = True
        

    print(f"Malicious/Suspicious score: {score}")
    print((det_stats["malicious"] + det_stats["suspicious"]) ,"/",tot_det)
    print(det_stats)

    report["n_engines"] = tot_det
    report["malicious"] = det_stats["malicious"]
    report["suspicious"] = det_stats["suspicious"]

    return analysis, det_stats, report



def get_file_info(file_path, api_key):

    client = vt.Client(api_key)

    # get info of a file
    with open(file_path, "rb") as f:
        hash = hashlib.sha256(f.read()).hexdigest()
    file_info = client.get_object("/files/"+hash)

    client.close()
    return file_info



if __name__ == "__main__":

    cfg_parser = ConfigParser()
    cfg_parser.read("config.conf")
    config = dict(cfg_parser["VIRUSTOTAL"])
    src = config["src"]
    print("Selected folder: %s" % src)

    api_key = config["api_key"]

    fnames = os.listdir(src)
    print("Number of files: %i" % len(fnames))

    if os.path.exists("./reports/vt_analysis_temp.pkl"):
        detection = pickle.load(open("./reports/vt_analysis_temp.pkl", "rb"))
        fnames = os.listdir(src)[len(detection["vt_detections"]):]
        print("State recovered.")
    else:
        detection = {
            "filename" : fnames,
            "vt_detections" : [],
            "n_engines" : []
        }
        print("New state created.")


    for file in tqdm(fnames):
        print("Scanning %s" % file)
        analysis, det_stats, short_stats = scan_file(os.path.join(src, file), api_key=api_key)
        detection["vt_detections"].append(short_stats["malicious"] + short_stats["suspicious"])
        detection["n_engines"].append(short_stats["n_engines"])
        pickle.dump(detection, open("./reports/vt/vt_analysis_temp.pkl", "wb"))


    # Report creation
    if not os.path.exists("./reports/"):
        os.mkdir("./reports/")
    df_vt = pd.DataFrame(detection)
    df_vt.to_csv(f"./reports/vt_analysis_{int(time.time())}.csv", index=False)
    os.remove("./reports/vt/vt_analysis_temp.pkl")