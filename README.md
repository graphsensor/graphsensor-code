# GRAPHSENSOR

## Prepare datasets
Two public datasets are used in this study:
[Sleep-EDF-20](https://gist.github.com/emadeldeen24/a22691e36759934e53984289a94cb09b),
[WISDM](https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+)

After downloading the datasets, the data can be prepared as follows:

Sleep-EDF-20:
```
`cd prepare_datasets`
python prepare_physionet.py --data_dir /path/to/PSG/files --output_dir edf_20_npz --select_ch "EEG Fpz-Cz"
```

WISDM:
```
save the raw accelerometer sensor data collected from the smartwatch to ./prepare_datasets/WSIDM/watch
```

## Training GRAPHSENSOR

Sleep-EDF-20:

The `config.json` file is used to update the training parameters.
To perform the standard K-fold crossvalidation, specify the number of folds in `config.json` and run the following:
```
chmod +x batch_train.sh
./batch_train.sh 0 /path/files
```
where the first argument represents the GPU id.

If you want to train only one specific fold (e.g. fold 0), use this command:
```
python3 train_Kfold_CV.py --device 0 --fold_id 0 --np_data_dir /path/to/npz/files
```

WSIDM:

```
python3 WISDM_train.py --model "GRAPHSENSOR"
python3 WISDM_train.py --model "MOBILENET"
python3 WISDM_train.py --model "RESNET"
python3 WISDM_train.py --model "EFFICIENTNET"
```

## Results
The log file of each fold is found in the fold directory inside the save_dir.  