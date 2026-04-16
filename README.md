#  Gradient Adversarial Manipulation of Executables for EXplainability-based Evasion (GAME4EXE)


The repository contains code referred to the work:

### Adversarial Malware Can Be Both Evasive and Deceiving: a Gradient-based Attack Against Prediction and Explainability in Windows PE Malware Detection

Please cite our work if you find it useful for your research and work.


## Code requirements
The code relies on the following python3.9+ libs.
Packages needed are:
* torch 2.7.1(+cu128 for CUDA)
* pefile 2024.8.26
* numpy 1.26.4
* pandas 2.2.3
* plotly 6.3.1
* tqdm 4.60.0
* vt-py 0.21.0


## How to use
Repository contains scripts of all experiments included in the paper:
Modify the arguments inside the config.conf file, if necessary, before running any experiment.
### Arguments
[CONFIG] - General config
* mode: optimization modes (M1 only prediction, M2 only explanation, M3 combined)
* malware_path: malware input folder path
* target_path: the target goodware path
* output_folder: output folder where the adversarials will be saved
* n_steps: number of optimization steps
* patience: early stopping patience
* lr: optimizer learning rate
  
[BOXPLOT_SINGLE] - For boxplot of a single generation report
* report_path = path to the csv report file
  
[BOXPLOT_MULTIPLE] - For boxplot of the three generation reports
* m1/m2/m3_report_path = path to the csv report files
  
[VIRUSTOTAL] - For the file submission and analysis to VirusTotal
* api_key: the VirusTotal API key
* src: the file folder to scan
  

### Scripts descriptions 
* __run_experiment.py__ : script to run the adversarial malware creation and transferability test. 
* __make_boxplot_xaimse_single.py__ : for creating boxplot of a single generation report (modify the argument report_path inside config.conf accordingly).
* __make_boxplot_xaimse_multiple.py__ : for creating boxplots of the three generation reports, one for each config (modify the arguments m1/m2/m3_report_path inside config.conf accordingly).
* __make_bytexaimse_plot.py__ : for creating the average ByteXAIMse plot
* __virustotal_analysis.py__ : for submitting and analyzing malware/adversarial files to VirusTotal (requires VT API key)

### How to run: execute the script followed by config filename (as shown in the following example)
python run_experiment.py config.conf

## Data
SHA256 hashes of the Windows PE files used in the experiment are reported in the files malwares_samples_sha256.txt and the respective target goodware sha256 for each model under the "samples" folder.  

## License
This work is under Attribution-NonCommercial-ShareAlike 4.0 International license. More information on LICENSE.md.











