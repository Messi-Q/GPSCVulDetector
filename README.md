# GPSCVulDetector

This repo is a python implementation of smart contract vulnerability detection of our method (CEG). 
Here, we explore using graph neural networks and expert knowledge for smart contract vulnerability detection.

## Requirements

### Required Packages
* **python**3 or above
* **TensorFlow** 2.0
* **sklearn** for model evaluation

Run the following script to install the required packages.
```shell
pip install --upgrade pip
pip install tensorflow==2.0
pip install scikit-learn
```

### Dataset

#### source code 
Original smart contract source code:

Ethereum smart contracts:  [Etherscan_contract](https://drive.google.com/open?id=1h9aFFSsL7mK4NmVJd4So7IJlFj9u0HRv)

Vntchain smart contacts: [Vntchain_contract](https://drive.google.com/open?id=1FTb__ERCOGNGM9dTeHLwAxBLw7X5Td4v)


#### Dataset structure in this project
Here, we present the dataset structure in our project, including the graph feature and pattern feature.

```shell
${GPSCVulDetector}
├── data
│   ├── loops
│   ├── timestamp
│   └── reentrancy
├── graph_feature
│   ├── loops
│   ├── timestamp
│   └── reentrancy
└── pattern_feature
    ├── feature_by_fc
    ├── feature_by_zeropadding
    └── label_by_extractor
```

**Note:** 
The graph feature of related smart contract is extracted by our [previous methods](https://github.com/Messi-Q/GNNSCVulDetector) published on the IJCAI 2020.
The pattern feature of related smart contract is extracted by the tools in the category `pattern_extractor`.  


## Running Project
* To run program, use this command: python GPSCVulDetector.py.
* Also, you can use specific hyper-parameters to train the model. All the hyper-parameters can be found in `parser.py`.

Examples:
```shell
python GPSCVulDetector.py
python GPSCVulDetector.py --model EncoderConv1D --lr 0.002 --dropout 0.2 --epochs 50 --batch_size 32
```

## References
1. Smart Contract Vulnerability Detection Using Graph Neural Networks. IJCAI 2020. [GNNSCVulDetector](https://github.com/Messi-Q/GNNSCVulDetector).
```
@inproceedings{ijcai2020-454,
  title     = {Smart Contract Vulnerability Detection using Graph Neural Network},
  author    = {Zhuang, Yuan and Liu, Zhenguang and Qian, Peng and Liu, Qi and Wang, Xiang and He, Qinming},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization}, 
  pages     = {3283--3290},
  year      = {2020},
}

```
