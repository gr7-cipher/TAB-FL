# Data

The datasets are not included in this repository. Download them from their providers and
place the files as follows:

```
data/raw/ciciot/Merged01.csv ... Merged04.csv     # CICIoT2023, CSV/MERGED_CSV
data/raw/toniot/Train_Test_IoT_*.csv               # TON_IoT, Train_Test_datasets/Train_Test_IoT_dataset (7 files)
```

- CICIoT2023: Canadian Institute for Cybersecurity, University of New Brunswick,
  https://www.unb.ca/cic/datasets/iotdataset-2023.html
- TON_IoT: UNSW Canberra Cyber Range Lab, https://research.unsw.edu.au/projects/toniot-datasets

Then build the processed arrays (class-capped CICIoT2023 sample, TON_IoT union feature
space, fixed train/test split):

```bash
cd code
python data_loader.py --raw ../data/raw --out ../data/processed
```

This writes `data/processed/ciciot2023.npz` and `data/processed/toniot.npz`, which all
experiment scripts read.
