# Intelligent Systems Project

This project regards the dataset processing and death classification of patients in the Septic Cardiomiopathy (SCM) database.

### Folder structure

The directory structure of your new project looks like this (please adjust the structure and its description to best fit your project): 

```
├── README.md          <- The top-level README for contributers of this project.
│
├── data
│   ├── raw            <- The original, immutable data dump.
│   └── processed      <- The final, canonical data sets for modeling.
│
├── requirements.txt   <- The file with instructions for reproducing the project environment (software).  
|                         Indicates the version of  software you are using.
│
├── code                <- Source code for use in this project.
|     │
|     ├── feature_selection  <- Functions that regard feature selection tasks in the project.
|     | 
|     ├── preprocessing <- Functions to load dataset and process the data.
|     │
|     ├── model         <- Functions to create models, run models, optization algorithms, etc.
|     │
|     ├── clustering <- Scripts and functions for visualizations.
|     │
|     ├── utils         <- Imports for the packages used.
|     |
|     ├── main          <- Main notebook to run the code.
|     |     
|     ├── graphs        <- Functions to create graphs used on the report.
|
|
├── report              <- Project report

```


### How to run the code

In order to run the project, the main.ipynb notebook should be used, which performs the entire project calling the other functions defined elsewhere.
