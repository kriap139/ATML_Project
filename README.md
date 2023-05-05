# Project Title

Source code for ATML Project. 

## Installation

1. Clone the repository
git clone https://github.com/username/repository.git
cd ATML_Project/

2. Install the dependencies
./scripts/install-mmdet.sh

3. Replace data folder with the provided data folder

## Usage

1. For traning run:
./scripts/train.sh src/train_config.py

2. Copy the train_config.py and epoch_x.pth to the data/models/v1-100/ directory, from the workdirs directory, or change the paths bellow.

3. Run test on test data
python src/test.py --model data/models/v1-100/epoch_x.pth  data/models/v1-100/train_config.py

4. Run test on train data
python src/test.py --model data/models/v1-100/epoch_x.pth --test-dataset train  data/models/v1-100/train_config.py
