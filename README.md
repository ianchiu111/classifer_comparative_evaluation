# Welcome
This project conduct a comparative evaluation of multiple classifiers on [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data).

## 📚 Data Schema
|Variable |	Definition | Key |
| --- | --- | --- |
|survival | Survival | 0 = No, 1 = Yes | 
|pclass | Ticket class | pclass: A proxy for socio-economic status (SES)1 = 1st = Upper, 2 = Middle, 3 = Lower |
|sex | Sex	| |
|Age | Age in years	| |
|sibsp | # of siblings / spouses aboard the Titanic | |
|parch | # of parents / children aboard the Titanic	| |
|ticket | Ticket number	| |
|fare | Passenger fare	| Ticket price|
|cabin | Cabin number | The cabin number assigned to the passenger, typically indicating the deck (letter) and room number. Many values are missing due to incomplete records.| 
|embarked | Port of Embarkation | The port where the passenger boarded the Titanic: C = Cherbourg, Q = Queenstown, S = Southampton. |


## 🔧 Classfier Practices
1. SVM
2. Random Forest
3. Gradient Boosting
4. XGBoost
5. Neural Network

## 📝 Learning Notes
### SVM
### Random Forest
1. Concepts: Randomly select features as if-then filters. 
2. Tree Construction: Bootstrap Bagging
3. Hyper-Parameters:
    - n_estimators: How many trees to process when training
    - max_depth: How deep does the tree
    - min_samples_split: The minimum number of training data (rows) required when if-then node processing
    - min_sample_leaf: The minimum number of training data (rows) required when in **leaf**
        - (In theory) min_samples_split = 2 *  min_samples_leaf
4. Scenario Discussion:
    - **High Noise** in Training Data (rows)
        - High max_depth may cause **Overfitting**
    - **Inference time linearly increases** with the increase of n_estimators

### Gradient Boosting
> Famous person: Friedman
#### (一) GBDT
1. Concepts: All **training data (rows)** and features selected in every tree.
2. Issues: It's the most likely to occue overfitting

#### (二) Stochastic Gradient Boosting
1. Concepts: Add **Row Subsampling** in GBDT

### XGBoost
#### Reference
1. [Classification using XGBoost in Python](https://www.educative.io/answers/classification-using-xgboost-in-python)
---
#### Notes
1. Concepts: Combine the random-selected concept of random forest to optimize gradient boosting 
2. Tree Construction: **Row/Column** Subsampling
3. Hyper-Parameters:
    - n_estimators: How many trees to process when training
    - max_depth: How deep does the tree
    - learning_rate: usually test 0.001 in the begining
    - min_child_weight: It should be a **Minimum threshold** for the **sum of Hessians (second-order gradients) in a leaf**.
    - sub_sample: 
        - It's **Row sampling**
        - Randomly selects **a fraction of the training data (rows)** for each tree. 
        - Helps reduce overfitting and adds randomness like in Random Forest.
    - colsample_bytree: 
        - It's **Column sampling per tree**
        - Randomly selects **a fraction of features once for the whole tree**.
        - Each split in that tree can only use those selected features.
    - colsample_bylevel
        - It's **Column sampling per tree level**
        - Randomly selects **a fraction of features separately at each depth level of the tree**.
    - colsample_bynode
        - It's **Column sampling per split (per node)**
        - Randomly selects **a fraction of features for every split.**

### Neural Network


