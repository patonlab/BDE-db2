 # BDE and BDFE predictions of Halogenated Species 

Models: Contains all GNN model as shown in the paper where Model 1 is the initial model, Model 2 is built with additional molecules involving multiple halogen heterocycles, and Model 3 is the final model which accounts for polyhaloalkyl molecules.  

Datasets: All BDE and BDFE datasets used in developing the models, testing the models. This folder is further organised based on datasets used for iterative training and testing. We also have the dataset for external validation provided.



## 1. Environment for BDE prediction
Create and activate the environment. All required python packages are wrapped in this `2D.yml` file. 

```
cd Example-BDE-prediction/
conda env create -f 2D.yml -n bde
conda activate bde
```

## 2. Running BDE prediction

The `Example-BDE-prediction/` folder contains an example notebook `test-prediction.ipynb` where the BDE model can be loaded and utilized for BDE prediction. The SMILES of the molecules can be provided as list to the prediction model. 


## 3. Citation
```
@article{
  title={Expansion of Bond Dissociation Prediction with Machine Learning to Medicinally and Environmentally Relevant Chemical Space}, 
  author={S. V., S. S.; Kim, K.; Kim, S.; St. John, P.; Paton, R. S.,},
  journal={ChemRxiv},
  year={2023}
}
To Do: To be updated with DOI and journal.
