# Addressing Recent Spikes of Severity of Wildfires in Northern California With Data Analysis and Machine Learning
This project analyzes environmental variables to model and predict wildfire risk, with the goal of informing land management decisions. For DATA399 Environmental Data Science taught by Professor Meunier.

## Authors 
- Jenna Hopkins - Willamette University, 3 + 1 MS Data Science Program, Data Science & Computer Science Major
- Jerrick Little - Willamette University, 3 + 1 MS Data Science Program, Data Science Major
- Anujan Tennathur - Willamette University, Data Science & Computer Science Major

## Overview
Repeated large-scale fire suppression without sufficient prevention measures has contributed to the accumulation of fuel understory in high-risk wildfire areas, increasing the severity of future wildfires. This risk is further intensified by anthropogenic climate change, which is driving drier atmospheric conditions and more flammable fuels. Contemporary wildfire detection and prediction has evolved from human observation to systems that use satellite imagery, ground-based sensors, and aerial surveillance. Despite these advancements, federal policy is still lagging behind modern-day research in regards to wildfire prevention and detection. 
Wildfires in northern California are increasing in frequency, severity, and economic impact, with annual losses estimated between $394 and $893 billion, including property damage, health impacts, and suppression costs. Climate models project continued increases in temperature and fuel aridity due to anthropogenic climate change, which will further intensify wildfire activity. In addition, long-term fire suppression has contributed to fuel accumulation, increasing the likelihood of high-severity fire events. As a result, wildfire risk is growing while current mitigation efforts remain inconsistently aligned with environmental conditions that drive fire severity. 
Two solutions are developed to address inefficient fuel treatment placement in northern California. The first is a spatial wildfire severity risk model that uses environmental predictors to identify areas with high likelihood of severe fire, producing interpretable risk surfaces that generalize to new fire events. This improves allocation of limited resources by prioritizing high-risk locations. The second is a fire regime classification framework that identifies distinct ecological and climatic fire environments and links them to appropriate treatment strategies. This improves treatment effectiveness by aligning interventions with local fuel and environmental conditions. Together, these solutions address both where treatments should be placed and how they should be implemented.
It is recommended that land management agencies transition from historically driven treatment strategies to a risk-based framework. This approach combines the predictive spatial model from Solution 1 to identify high-risk areas with the regime-based framework from Solution 2 to guide treatment selection within those areas. This integrated system improves the efficiency and effectiveness of fuel treatments by ensuring that both placement and treatment type are aligned with environmental fire risk.

## Data Sources
* MTBS (Monitoring Trends in Burn Severity)
* CAL FIRE (California Department of Forestry and Fire Protection)
* LANDFIRE (Landscape Fire and Resource Management Planning Tools)
* PRISM (Parameter-elevation Regressions on Independent Slopes Model)

Raw data files are too big to be stored on GitHub, and can be found in this Google Drive folder: https://drive.google.com/drive/folders/1ywLNy2Vk8i748Z1bfy3FXhVBVB3FLQhz?usp=sharing 

## Methods
We applied the following techniques:

- Data joining in Python -> combined_dataset.csv
- Dataset preprocessing in R
- Normality distribution tests
    - Shapiro test
    - Anderson-Darling test
- Testing for differences in severity across fires
    - Kruskall-Wallis test
    - Dunn's test
- Forming fire group clusters
    - Hierarchial Cluster Analysis
    - Elbow Method
- Unsupervised machine learning
    - NMDS
    - UMAP
    - HBDSCAN
- Supervised machine learning
    - Random Forest

## Key Findings

## Repository Structure
|--- AnalysisDatasets/
|    |--- combined_dataset.csv
|    |--- new_fiers_dataset.csv
|--- Python Code (joining source data)/
|    |--- join_dataset.py
|    |--- new_data.py
|--- R Code (analysis)/
|    |---
|--- .gitignore
|--- README.md

## How to Run
1. Clone the repository
2. Run script:
   - `AJ_CODE_RMD.Rmd`

## White Paper
