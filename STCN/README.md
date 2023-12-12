# Test-time training using a offline-trained STCN model
This code is based on the 
[STCN](https://github.com/hkchengrex/STCN) repository.

## Installation


You can find below the installation script:

[//]: # (<details>)

[//]: # (  <summary> <b> Code </b> </summary>)

```
python -m venv ENV
source ENV/bin/activate
pip install torch torchvision
pip install pyyaml
```

[//]: # (</details>)

## Datasets

We evaluated our test-time training strategy on four datasets:
* [DAVIS 2017](https://davischallenge.org/)
* [YOUTUBE-VOS 2018](https://youtube-vos.org/)
* [MOSE](https://henghuiding.github.io/MOSE/)
* [DAVIS-C](https://jbertrand89.github.io/test-time-training-vos/)

Please see in [DATA_PREPARATION](https://github.com/ttt-matching-based-vos/ttt_matching_vos/blob/main/DATA_PREPARATION.md) 
to download the datasets.


## Pre-saved configs

The test-time training strategy can be run on top of any STCN model trained offline. In our study, we evaluated on top 
of the three [models](https://github.com/hkchengrex/STCN/releases/tag/1.0) provided in the original repository.
* `stcn_s01`, model trained  without real videos (model used in the sim2real transfer study)
* `stcn`, model trained with real videos (model used in the corrupted examples study)
* `stcn_s0`, model trained with static videos

For each model, you can run the test-time training strategy using three losses:
* the `tt-mcc` loss, our proposed method, using temporal information through the mask cycle consistency
* the `tt-ae` loss, an auto-encoder-based variant
* the `tt-ent` loss, an entropy-based variant

For each combination of an offline trained model and a loss, we saved the parameters used to run our test-time training 
strategy in a configuration file, saved in the [`ttt_configs`](https://github.com/jbertrand89/test_time_learning/tree/main/STCN/ttt_configs) folder.

You can save your own config by running:
```
python ttt/config/save_config.py --config_name <config_name> <your parameters list>
```

## Run your test-time training strategy

We provide below the scripts to run the sim2real transfer study on the four datasets.
```
python eval_all_datasets_ttt.py --config_name stcn_s01_mcc --dataset_name davis --split val --dataset_dir $DATA_DIR --output_dir $OUTPUT_DIR --seed $SEED
python eval_all_datasets_ttt.py --config_name stcn_s01_mcc --dataset_name youtube --split valid --dataset_dir $DATA_DIR --output_dir $OUTPUT_DIR --seed $SEED
python eval_all_datasets_ttt.py --config_name stcn_s01_mcc --dataset_name mose --split valid --dataset_dir $DATA_DIR --output_dir $OUTPUT_DIR --seed $SEED
```

We also provide the scripts to run the corrupted examples analysis below.
```
python eval_all_datasets_ttt.py --config_name stcn_mcc --dataset_name davis --split val --dataset_dir $DATA_DIR --corrupted_image_dir $CORRUPTED_IMAGE_DIR --output_dir $OUTPUT_DIR --seed $SEED
```

To run test time training, you need to select your configuration (depending on the offline trained STCN model and the 
test-time training loss used) and the dataset you want to run it for.

### Choose your configuration file

Choose your configuration file in the [ttt_configs](STCN/ttt_configs) folder (it can be one of your own as well).
For our main study, we used:
* `stcn_s01_mcc.yaml` (sim2real)
* `stcn_mcc.yaml` (corrupted examples)

### Choose your dataset

For each dataset, you will need to specify the following parameters:
* DATASET_NAME: between `davis`, `youtube`, `mose` or `davis-c`
* DATASET_DIR: the root directory where your dataset is saved. For example for the DAVIS-2017 dataset, it will be `<your_dataset_folder>/DAVIS/2017`
* CORRUPTED_IMAGE_DIR: the root directory containing the RGB frames with corruption. For example, for medium strength of
the brighness corruption, it will be <your_dataset_folder>/DAVIS-C/med/brightness.
* SPLIT: the name of the split you want to test on (for our main study we used the `val` split for DAVIS, and the `valid` 
split for YOUTUBE and MOSE)
* OUTPUT_DIR: the name of the output directory where to save the predicted masks


### Additional parameters

To run multiple model and evaluate the mean and standard deviation, we ran the models for multiple seeds. We define an
additional parameter:
* SEED: the seed used (experimented with 1 / 5 / 10)





