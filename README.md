# MALDI-image-representation
Code Repository for "Tailored deep learning strategy improves MALDI-MS image representation for metabolite identification in tumor tissue", 2026, Annalena Weissert, et. al. (working title)

This repository contains the code for preprocessing, training, and evaluation used in the associated paper. A small example dataset is included to test the workflow.

## FOLDER STRUCTURE
├── ExampleData_Labels/\
├── ExampleData_Processed/\
├── ExampleData_Raw/\
├── ExampleData_Training/\
├── TrainingOutput/\
├── 0_Data_raw_to_processed.ipynb\
├── 1_Data_processed_to_training.ipynb\
├── 2_Training_BYOL.py\
├── 3_Evaluation.ipynb\
└── AugmentationCuda.py\

## DATA
All folders starting with 'ExampleData' contain a small example dataset. This dataset is a reduced subset of the original data, with fewer m/z-values and smaller image sections.
The full processed dataset used in the paper is published under the CC-BY 4.0 license and is available here: https://doi.org/10.17877/RTG2624-2026-MRKM1K1K

## WORKFLOW
The scripts are intended to be run in numerical order.
- If you use the included example data, start with: 0_Data_raw_to_processed.ipynb
This notebook converts the raw example data from "ExampleData_Raw" into processed data in "ExampleData_Processed".
- Then, or if you start with the real data, run: 1_Data_processed_to_training.ipynb
This notebook converts the processed data (Example or Real) into the training dataset.
- Next, 2_Training_BYOL.py performs the model training. The best model weights and training progress files, including text files and images, are saved in 'TrainingOutput'.
- Finally, run 3_Evaluation.ipynb. This notebook shows how to load trained model weights and evaluate the results using the example data.
- The file 'AugmentationCuda.py' contains the code for the custom augmentation pipeline. It is used by 2_Training_BYOL.py and usually does not need to be modified.

### TRAINED MODEL WEIGHTS
The folder 'TrainingOutput/ModelParams/' contains the trained model weights from the full training run used to generate the results reported in the paper.
'ResNet50_FullBYOL-default_training-21_01_2026.pt' are the weights for training with default augmentations. Analougusly, 'ResNet50_FullBYOL-custom_training-19_01_2026.pt' the weights for custom augmentations.

### FURTHER EXPLANATION / DETAILS
More detailed explanations are provided as comments in the code. Places where file names, paths, or training parameters may need to be adapted are marked with #!

## Installation
It is recommended to use a virtual Python environment. On Linux, run:\
python -m venv .venv\
source .venv/bin/activate\
python -m pip install -r requirements-venv.txt\

Python Version used for training and evaluation: python 3.13\
**Note:** byol_pytorch 0.8.2 was used for training the weights used in the paper. The newer version (0.9.1), isn't compatible with the format of the saved weights.


## CITATION
If you use this code or the associated dataset, please cite the corresponding paper and dataset publication.\
Paper (working title):\
Weissert, Annalena; Begher-Tibbe, Brigitte; Reinders, Jörg; Edlund, Karolina; Myllys, Maiju; Glotzbach, Annika; Lücke, Simon; Marchan, Rosemarie; Hengstler, Jan G.; Rahnenführer, Jörg, 2026, "Tailored deep learning strategy improves MALDI-MS image representation for metabolite identification in tumor tissue"

Dataset: \
Weissert, Annalena; Begher-Tibbe, Brigitte; Reinders, Jörg; Edlund, Karolina; Myllys, Maiju; Glotzbach, Annika; Lücke, Simon; Marchan, Rosemarie; Hengstler, Jan G.; Rahnenführer, Jörg, 2026, "Preprocessed MALDI-MS imaging dataset across tumor stages in a murine breast cancer model", https://doi.org/10.17877/RTG2624-2026-MRKM1K1K

## LICENSE:
This work is licensed under MIT license: 

Copyright (c) 2026 Annalena Weissert

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
