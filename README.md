# GPSCVulDetector

This repo is a python implementation of combining graph neural network with expert patterns for smart contract vulnerability detection. 

## Requirements

### Required Packages
* **python**3 or above
* **TensorFlow** 2.0
* **numpy** 1.18.2
* **sklearn** for model evaluation

Run the following script to install the required packages.
```shell
pip install --upgrade pip
pip install tensorflow==2.0
pip install numpy==1.18.2
pip install scikit-learn
```

### Dataset
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
    ├── feature_by_zeropadding
    ├── original_pattern_feature
    └── label_by_autoextractor
```

## Graph extractor & Pattern extractor
1. **Graph:** The contract graph and its feature are extracted by the automatic graph extractor 
implemented by our [previous methods](https://github.com/Messi-Q/GNNSCVulDetector) and in this directory `graph_extractor_example`.
2. **Pattern:** The expert pattern and its feature are extracted by the automatic pattern extractor in this directory `pattern_extractor_example`.  


## Running Project
* To run program, use this command: python GPSCVulDetector.py.
* Also, you can use specific hyperparameters to train the model. All the hyperparameters can be found in `parser.py`.

Examples:
```shell
python GPSCVulDetector.py
python GPSCVulDetector.py --model CGE --lr 0.002 --dropout 0.2 --epochs 50 --batch_size 32
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
2. Towards Automated Reentrancy Detection for Smart Contracts Based on Sequential Models. IEEE Access. [ReChecker](https://github.com/Messi-Q/ReChecker).
```
@article{qian2020towards,
  title={Towards Automated Reentrancy Detection for Smart Contracts Based on Sequential Models},
  author={Qian, Peng and Liu, Zhenguang and He, Qinming and Zimmermann, Roger and Wang, Xun},
  journal={IEEE Access},
  year={2020},
  publisher={IEEE}
}
```