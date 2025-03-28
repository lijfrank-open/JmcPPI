# JmcPPI
The official implementation of the paper "[Joint Masked Reconstruction and Contrastive Learning for Mining Interactions Between Proteins](https://arxiv.org/abs/2503.04650)", which has been accepted by.  
Authors: Jiang Li, Xiaoping Wang  
Affiliation: Huazhong University of Science and Technology (HUST)  

## Citation
```bibtex
@article{li2025tracing,
    title={Joint Masked Reconstruction and Contrastive Learning for Mining Interactions Between Proteins},
    author={Jiang Li and Xiaoping Wang},
    year={2025},
    journal = {},
    volume = {},
    number={},
    pages = {1-22},
    doi={}
}
```

## Requirement
Checking and installing environmental requirements
```python
pip install -r requirements.txt
```

## Run
### SHS27k
```bash
CUDA_VISIBLE_DEVICES=2 python -u train.py --dataset SHS27k --split_mode partition_scheme --ckpt_path ../rec_trained_model/rec_model_SHS27k_128.ckpt
```
### SHS148k
```bash
CUDA_VISIBLE_DEVICES=2 python -u train.py --dataset SHS148k --split_mode partition_scheme --ckpt_path ../rec_trained_model/rec_model_SHS148k_128.ckpt
```
### STRING
```bash
CUDA_VISIBLE_DEVICES=2 python -u train.py --dataset STRING --split_mode partition_scheme --ckpt_path ../rec_trained_model/rec_model_STRING_128.ckpt
```
Here, `partition_scheme` is one of the three datasets (Random, BFS, and DFS).

## Dataset
The raw data of SHS27k, SHS148k, and STRING can be downloaded from [Google Drive](https://drive.google.com/file/d/1hJVrQXddB9JK68z7jlIcLfd9AmTWwgJr/view?usp=sharing):

* `protein.STRING.sequences.dictionary.tsv`      Protein sequences of STRING
* `protein.actions.STRING.txt`     PPI network of STRING
* `STRING_AF2DB`     PDB files of protein structures predicted by AlphaFold2

Pre-process raw data to generate feature and adjacency matrices (also applicable to any new dataset):
```
python ./raw_data/data_process.py --dataset data_name
```
where `data_name` is one of the three datasets (SHS27k, SHS148k, and STRING).