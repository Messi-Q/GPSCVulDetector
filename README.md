# GPSCVulDetector

This repo is a python implementation of combining graph neural network with expert knowledge for smart contract vulnerability detection. 

## Citation
Please use this citation in your paper if you refer to our [paper](https://ieeexplore.ieee.org/abstract/document/9477066/) or code.
```
@article{liu2023combining,
  title={Combining Graph Neural Networks With Expert Knowledge for Smart Contract Vulnerability Detection},
  author={Liu, Zhenguang and Qian, Peng and Wang, Xiaoyang and Zhuang, Yuan and Qiu, Lin and Wang, Xun},
  journal={IEEE Transactions on Knowledge \& Data Engineering},
  volume={35},
  number={02},
  pages={1296--1310},
  year={2023},
  publisher={IEEE Computer Society}
}
``` 

## Requirements

### Required Packages
* **python** 3+
* **TensorFlow** 2.0
* **numpy** 1.18.2
* **sklearn** 0.20.2

Run the following script to install the required packages.
```shell
pip install --upgrade pip
pip install tensorflow==2.0
pip install numpy==1.18.2
pip install scikit-learn==0.20.2
```


## Graph extractor & Pattern extractor
1. **Graph:** The contract graph and its feature are extracted by the automatic graph extractor in the `graph_extractor_example` directory (or refer to our [previous methods](https://github.com/Messi-Q/GNNSCVulDetector)).
2. **Pattern:** The expert pattern and its feature are extracted by the automatic pattern extractor in the `pattern_extractor_example` directory.  


Notably, you can also use the features extracted in [AMEVulDetector](https://github.com/Messi-Q/AMEVulDetector).

If any question, please email to messi.qp711@gmail.com.


## Running Project
* To run program, please use this command: python3 GPSCVulDetector.py.
* Also, you can set specific hyperparameters, and all the hyperparameters can be found in `parser.py`.

Examples:
```shell
python3 GPSCVulDetector.py
python3 GPSCVulDetector.py --model CGE --lr 0.002 --dropout 0.2 --epochs 100 --batch_size 32
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
