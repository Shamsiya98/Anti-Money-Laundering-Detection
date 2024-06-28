# Anti-Money-Laundering-Detection

This repo provides model training of Graph Attention Network and Machine learning models in Anti Money Laundering Detection problem.  
Dataset: https://drive.google.com/drive/folders/1XL0ZrURBWGfiqr8xWTYKXS4EPYVPpQWi?usp=drive_link 

## Usage
Please create the corresponding folder before you run the script. 
Put the .csv file into raw folder, [dataset.py](dataset.py) will create "processed" folder with processed data once you run the [train.py](train.py).  
Make sure the directories are created as below:

```bash
├── data
│   ├── raw
├── results
├── dataset.py
├── GNN.py
├── GML.py
└── train.py
```
The detailed report of this project can be assessed here: https://drive.google.com/file/d/1cEevy5udRVQRkPaxwkTMgQJlzbDH3YTl/view?usp=sharing
