These scripts were utilized to generate the results in the recent TASLP submission.

# Generate data
To run the scripts, first set the variable in the `paths.env` file.

The data generation requires the TIMIT dataset and Surrey RIRs dataset (not provided here).

Also, set `config.yaml` for generating the 'train', 'val' and 'test' data. The run the following for each case:

      python prepare_input_data.py

# Train models

To train the models, select either 'MLP' or 'CNN' in the `config.yaml`, then run

      python main.py
      

# Evaluate models

The folder `test_scripts` contains the code which was used to evaluate the networks outputs. These scripts are slightly older than those in the main folder, they will require some adaptation.

To run the evaluation, do as follows:

      cd test_scripts
      
You need to modify the `project_path` (here set to `/vol/vssp/mightywings`) where required in the scripts, and the `matlab_path` as well, if present at all.
You will also need to download the pre-trained models (link below) in the `project_path/B_format/RESULTS/models/Results` path.

To test the model, generate the separated audio, and evaluate the SNR-based metrics and PESQ, execute

      python run_test.py B_format train 12BB01 12BB01 ['theta','MV'] '' 'newnorm'
      
for each model. The syntaxt was adapted to the pre-trained files thus, in case you wish to run your own trained models, you will need to modify it accordingly.

To plot the results for four metrics, run

      python plot_metrics_allnet.py B_format train 12BB01 12BB01 ['theta','MV'] '' 'newnorm'
      

To generate the results for the word accuracy, run

      python evaluate_sr.py B_format train 12BB01 12BB01 ['theta','MV'] '' 'newnorm'
      
and
      
      python plot_srnet.py B_format train 12BB01 12BB01 ['theta','MV'] '' 'newnorm'



#  Pre-trained models

https://zenodo.org/record/7427355#.Y5eLyS1Q1QJ
