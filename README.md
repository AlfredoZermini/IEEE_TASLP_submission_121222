### USAGE

# Generate data
Requires TIMIT dataset and Surrey RIRs dataset (not provided here).

(set config.yaml for 'train', 'val' and 'test')

      python prepare_input_data.py

# Train models

      python main.py
      

# Evaluate model

The folder `test_scripts` contains the code which was used to evaluate the networks outputs. These scripts are slightly older (and require some adaptation) than those in the main folder.      
To run the evaluation, do as follows:
      cd test_scripts



#  Pre-trained models

https://zenodo.org/record/7427355#.Y5eLyS1Q1QJ
