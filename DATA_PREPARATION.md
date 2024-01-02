# Data preparation

We evaluated our test-time training strategy on four datasets:
* [DAVIS 2017](https://davischallenge.org/)
* [YOUTUBE-VOS 2018](https://youtube-vos.org/)
* [MOSE](https://henghuiding.github.io/MOSE/)
* [DAVIS-C](https://jbertrand89.github.io/test-time-training-vos/)

```
DATASETS
├── DAVIS
│   └── 2017
│       ├── test-dev
│       │   ├── Annotations
│       │   └── ...
│       └── trainval
│           ├── Annotations
│           └── ...
├── DAVIS-C
│   │── high
│   │   ├── brightness
│   │   └── ...
│   │── med
│   │   ├── brightness
│   │   └── ...
│   └── low
│       ├── brightness
│       └── ...
├── MOSE
│   │── train
│   │   ├── Annotations
│   │   └── ...
│   └── valid
│       ├── Annotations
│       └── ...
└── YouTube2018
    ├── all_frames
    │   └── valid_all_frames
    └── valid
```

### Download the datasets

```
pip install gdown

python ttt_download_datasets.py --root_dir <your_dataset_root_folder>
```

[//]: # (## References)

[//]: # ()
[//]: # ([1] J. Pont-Tuset, F. Perazzi, S. Caelles, P. Arbeláez, A. Sorkine-Hornung, and L. Van Gool. )

[//]: # ([The 2017 davis challenge on video object segmentation.]&#40;https://arxiv.org/abs/1704.00675&#41;)

[//]: # ()
[//]: # ([2] N. Xu, L. Yang, Y. Fan, D. Yue, Y. Liang, J. Yang, and T. Huang. [Youtube-vos: A large-scale video object)

[//]: # (segmentation benchmark.]&#40;https://arxiv.org/abs/1809.03327&#41;)

[//]: # ()
[//]: # ([3] H. Ding,  C. Liu, S. He, X. Jiang, P.H.S. Torr. )

[//]: # ([MOSE: A New Dataset for Video Object Segmentation in Complex Scenes]&#40;https://openaccess.thecvf.com/content/ICCV2023/papers/Ding_MOSE_A_New_Dataset_for_Video_Object_Segmentation_in_Complex_ICCV_2023_paper.pdf&#41;)
