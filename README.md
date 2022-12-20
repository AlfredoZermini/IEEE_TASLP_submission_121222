These scripts were utilized to generate the results in the recent TASLP submission.

# Generate data
Requires TIMIT dataset and Surrey RIRs dataset (not provided here).

(set config.yaml for 'train', 'val' and 'test')

      python prepare_input_data.py

# Train models

      python main.py
      

# Evaluate model

The folder `test_scripts` contains the code which was used to evaluate the networks outputs. These scripts are slightly older than those in the main folder, they will require some adaptation.

To run the evaluation, do as follows:

      cd test_scripts
      
You need to modify the `project_path` (here set to `/vol/vssp/mightywings`) where required in the scripts, and the `matlab_path` as well, if present at all.
You will also need to download the pre-trained models (link below) in the `project_path/B_format/RESULTS/models/Results` path.

To test the model, generate the separated audio, and evaluate the SNR-based metrics and PESQ, execute:

      python run_test.py B_format train 12BB01 12BB01 ['theta','MV'] '' 'newnorm'
      
The syntaxt was adapted to the pre-trained files thus, in case you wish to run your own trained models, you will need to modify it accordingly.




#  Pre-trained models

https://zenodo.org/record/7427355#.Y5eLyS1Q1QJ
