1. Install the requirements on a Python 3.9 venv/conda enviroment
2. If necessary, modify the arguments inside the config.conf file. Arguments:
	mode: optimization modes (M1 only prediction, M2 only explanation, M3 combined)
	malware_path: malware input folder path
	target_path: the target goodware path
	output_folder: output folder where the adversarials will be saved
	n_steps: optimization steps
	patience: early stopping patience
	lr: optimizer learning rate
3. Run the script run_experiment.py for adversarial malwares generation and transferability test
4. For creating the XAIMse Boxplots:
	- Run the script make_boxplot_xaimse_single.py for creating boxplot of a single generation report (modify the argument report_path inside config.conf accordingly)
	- Run the script make_boxplot_xaimse_multiple.py for creating boxplots of the three generation reports, one for each config (modify the arguments m1/m2/m3_report_path inside config.conf accordingly)
5. For creating the ByteXAIMse plot run the script make_bytexai_mse_plot.py
6. For submitting files to Virustotal and creating a detection report run the script virustotal_analysis.py, requires an API key (modify the arguments api_key and src accordingly for setting the VT API key and the file folder to scan)
	
