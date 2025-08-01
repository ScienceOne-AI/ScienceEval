---
license: apache-2.0
---
zinc250k contains the 250k molecule subset used in the "Automatic Chemical Design Using a 
Data-Driven Continuous Representation of Molecules" paper (doi:10.1021/acscentsci.7b00572).

The dataset contains the original columns from 
https://github.com/aspuru-guzik-group/chemical_vae/blob/main/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv,
namely smiles, logP, QED, and SAS and an additional selfies column. 

This dataset can be used for benchmarking chemical Language Models or training new ones on molecular property regression.

To download the dataset:
```
from datasets import load_dataset

dataset = load_dataset("edmanft/zinc250k")
```
All credits go to the authors of the paper and the code repository https://github.com/aspuru-guzik-group/chemical_vae/

