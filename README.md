# Test-time Training for Matching-based Video Object Segmentation

[[arXiv](https://openreview.net/pdf?id=9QsdPQlWiE) ] [[project page](https://jbertrand89.github.io/test-time-training-vos/)]

This repository contains official code for our NeurIPS 2023 paper 
[Test-Time Training for Matching-Based Video Object Segmentation](https://openreview.net/pdf?id=9QsdPQlWiE).

## What do we have here?
1. [Installation](#installation)

2. [Data preparation](#data-preparation)

3. [Test-time training](#test-time-training)

4. [Citation](#citation)

[//]: # (5. [References]&#40;#references&#41;)


## Installation

You can find below the installation script:

```
python -m venv ENV
source ENV/bin/activate
pip install torch torchvision
pip install pyyaml
```



## Data preparation

We evaluated our test-time training strategy on four datasets:
* [DAVIS 2017](https://davischallenge.org/)
* [YOUTUBE-VOS 2018](https://youtube-vos.org/)
* [MOSE](https://henghuiding.github.io/MOSE/)
* [DAVIS-C](https://jbertrand89.github.io/test-time-training-vos/)


For more details on the datasets, please refer to 
[DATA_PREPARATION](https://github.com/ttt-matching-based-vos/ttt_matching_vos/blob/main/DATA_PREPARATION.md).


## Test time training

We evaluated our proposed test-time training strategy starting from two offline-trained matching-based models:
* [STCN](https://github.com/hkchengrex/STCN)
* [XMem](https://github.com/hkchengrex/XMem)

### Test-time training with the STCN model

For more details for please refer to [STCN](STCN/README.md).


### Test-time training with the XMem model

For more details for please refer to [XMem](XMem/README.md).


## Citation
If you use this code for your research, please consider citing our papers:
```bibtex
@inproceedings{bertrand2023ttt_vos,
  title={Test-time Training for Matching-based Video Object Segmentation},
  author={Bertrand, Juliette and Kordopatis-Zilos, Giorgos and Kalantidis, Yannis and Tolias, Giorgos},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

## Acknowledgement

We want to thank [@hkchengrex](https://github.com/hkchengrex) 
for providing publicly available code and pretrained models for
[STCN](https://github.com/hkchengrex/STCN) and [XMem](https://github.com/hkchengrex/XMem).
