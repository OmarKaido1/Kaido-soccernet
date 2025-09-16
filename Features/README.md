# Extract Features from SoccerNet-v2

## Create conda environment

``` bash
python=3.8.10
cudnn cudatoolkit=10.1
pip install scikit-video tensorflow==2.3.0 imutils opencv-python==3.4.11.41 SoccerNet moviepy scikit-learn
```

## Extract ResNET features for all 550 games (500 + 50 challenge)

```bash
python E:\secodtst\Features\ExtractResNET_TF2.py --soccernet_dirpath "E:\E\soccernet\dataset" --back_end=TF2 --features=ResNET --video LQ --transform crop --verbose --split all
```

## Reduce features for all 550 games (500 games to estimate PCA + 50 challenge games for inference)

```bash
python Features/ReduceFeaturesPCA.py --soccernet_dirpath "E:\E\soccernet\dataset"
```

## Extract ResNET features for a given video

```bash
python E:\\secodtst\\Features\\VideoFeatureExtractor2.py --path_video "E:\dataset\england_epl\2014-2015\2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal\1_224p.mkv" --path_features "E:\secodtst\result" --start 0 --duration 5600 --overwrite --PCA E:\secodtst\Features\pca_512_TF2.pkl --PCA_scaler E:\secodtst\Features\average_512_TF2.pkl
```

