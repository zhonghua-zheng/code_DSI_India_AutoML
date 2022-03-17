# AutoML to identify feature importance for fine particle estimates over India

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Introduction](#introduction)
- [Scripts and Data](#scripts-and-data)
  - [Prerequisite](#prerequisite)
  - [Scripts](#scripts)
  - [Data](#data)
- [Acknowledgments](#acknowledgments)

<!-- /code_chunk_output -->


## Introduction

This repository is a supplementary to the manuscript **"Automated machine learning to evaluate the information content of tropospheric trace gas columns for fine particle estimates over India: a modeling testbed"**.

The objectives of this project are:

- Use [**pyEOF**](https://pyeof.readthedocs.io/en/latest/) to delineate regions for analysis
- Use **[FLAML](https://github.com/microsoft/FLAML)** to train the models (emulators) from the **[GEOS-Chem](https://acmg.seas.harvard.edu/geos_chem)** simulations using different combination of features
- Analysis the **feature importance** of different models

## Scripts and Data

### Prerequisite

- If you do not have the **"[conda](https://docs.conda.io/en/latest/)"** system

  ```bash
  # Download and install conda
  $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
  $ chmod +x Miniconda3-latest-Linux-x86_64.sh
  $ ./Miniconda3-latest-Linux-x86_64.sh
  # Edit .bash_profile or .bashrc
  PATH=$PATH:$HOME/.local/bin:$HOME/bin:$HOME/miniconda3/bin
  # Activate the conda system
  $source .bash_profile
  # OR source .bashrc
  ```

- Create and activate pyEOF environment: https://pyeof.readthedocs.io/en/latest/installation.html 

- Create and activate your own conda environment for flaml

  ```bash
  # Create an environment "partmc" and install the necessary packages
  conda env create -f environment.yml
  # Activate the "partmc" environment
  conda activate flaml
  ```

### Scripts

| Tasks                         | Folders or Files              | Fig or Tab in paper | Fig or Tab in preprint |
| ----------------------------- | ----------------------------- | ------------------- | ---------------------- |
| EOF/REOF to delineate regions | 1_get_regions_eof_reofs.ipynb |                     |                        |
| data preparation              | 2_automl/ml_data_prep/*       |                     |                        |
| model development (AutoML)    | 2_automl/automl/*             |                     |                        |
| analysis and figures          | 3_scripts_for_figures         | all figures         | all figures            |

### Data

- How to use the data?

  - Download the data

  - use the following commands to link your data

    ```bash
    cd code_DSI_India_AutoML
    mkdir data
    cd data
    # assume you have all your data in the folder "/Users/zhonghuazheng/data/", then
    ln -s /Users/zhonghuazheng/data/daily_meteo.nc daily_meteo.nc
    ln -s /Users/zhonghuazheng/data/daily_aod.nc daily_aod.nc
    ln -s /Users/zhonghuazheng/data/daily_gas_column.nc daily_gas_column.nc
    ln -s /Users/zhonghuazheng/data/daily_emission.nc daily_emission.nc
    ln -s /Users/zhonghuazheng/data/daily_surface_pm25_RH50.nc daily_surface_pm25_RH50.nc
    ln -s /Users/zhonghuazheng/data/land_mask.nc land_mask.nc
    ```

- raw data (from GEOS-Chem)

  | Num  | Folder or File             | Comments            | How to get it?                                               |
  | ---- | -------------------------- | ------------------- | ------------------------------------------------------------ |
  | 1.1  | daily_meteo.nc             | meteorological data |                                                              |
  | 1.2  | daily_aod.nc               | AOD data            | converted by [FlexAOD](http://pumpkin.aquila.infn.it/flexaod/) |
  | 1.3  | daily_gas_column.nc        | gas column          |                                                              |
  | 1.4  | daily_emission.nc          | emission data       |                                                              |
  | 1.5  | daily_surface_pm25_RH50.nc | surface PM2.5       |                                                              |
  | 1.6  | land_mask.nc               | land mask           |                                                              |
  
- data for machine learning

  | Num  | Folder or File         | Comments                                                     | How to get it?                                               |
  | ---- | ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | 2.0  | r_mask.nc              | mask for four regions                                        | data 1.5 and 1.6 and script 1_get_regions_eof_reofs.ipynb    |
  | 2.1  | c_r_daily_train.gzip   | daily training data for regions and the union of four regions | raw data, data 2.0 and script 2_automl/ml_data_prep/clusters_regions_daily_data_prep.ipynb |
  | 2.2  | c_r_daily_test.gzip    | daily testing data for regions and the union of four regions | raw data, data 2.0 and script 2_automl/ml_data_prep/clusters_regions_daily_data_prep.ipynb |
  | 2.3  | c_r_monthly_train.gzip | monthly training data for regions and the union of four regions | raw data, data 2.0 and script 2_automl/ml_data_prep/clusters_regions_monthly_data_prep.ipynb |
  | 2.4  | c_r_monthly_test.gzip  | monthly testing data for regions and the union of four regions | raw data, data 2.0 and script 2_automl/ml_data_prep/clusters_regions_monthly_data_prep.ipynb |
  | 2.5  | daily_train.gzip       | daily training data for all gridcells                        | raw data and script 2_automl/ml_data_prep/all_gridcell_daily_data_prep.ipynb |
  | 2.6  | daily_test.gzip        | daily testing data for all gridcells                         | raw data and script 2_automl/ml_data_prep/all_gridcell_daily_data_prep.ipynb |
  | 2.7  | monthly_train.gzip     | monthly training data for all gridcells                      | raw data and script 2_automl/ml_data_prep/all_gridcell_monthly_data_prep.ipynb |
  | 2.8  | monthly_test.gzip      | monthly testing data for all gridcells                       | raw data and script 2_automl/ml_data_prep/all_gridcell_monthly_data_prep.ipynb |

- data for analysis

  | Num  | Folder                               | Comments                                  | How to get it?                                               |
  | ---- | ------------------------------------ | ----------------------------------------- | ------------------------------------------------------------ |
  | 3.1  | 3_scripts_for_figures/data/*         | data from AutoML and a series of analysis | Note: plotting figure 1 needs the results from script 1_get_regions_eof_reofs.ipynb |
  | 3.2  | 3_scripts_for_figures/data/ranking/* | ranking from AutoML                       | 3_scripts_for_figures/_save_ranking_scores\*.ipynb           |

## Acknowledgments

- ExxonMobil Research and Engineering Company (EMRE)
- [Data Science Institute](https://datascience.columbia.edu/) (DSI) at Columbia University 
- Prof. [Ruth S. DeFries](http://www.ruthdefries.e3b.columbia.edu/ruth-defries/) and Prof. [Marianthi-Anna Kioumourtzoglou](http://www.publichealth.columbia.edu/people/our-faculty/mk3961)

