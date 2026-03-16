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


## 🔧 Classfier Testing
### Testing Notes
1. Use dataset with categorical but encoded data may cause error!
    - ⚠️ If the unique values in a specific categorical columns are not same between training and testing dataset. It will cause **Feature Mismatch issue**
2. When I remove the Nan value in testing dataset, the Correct Rate of models all decrease a little bit. 
3. PCA visualization is unavailable for categorical data, due to no numeric distance meaning.
    - Available: SVM, Random Forest
    - Unavailable: XGBoost

### Testing Results
1. SVM
    ```python
    ## data columns
    "Survived": int
    "Pclass": int
    "Age": int
    "SibSp": int
    "Parch": int
    "Fare": float

    ## model configs
    ### config explanations
    1. C: limitation of model complexity, to avoid over-fitting
    2. kernel: kernel function used in non-linear model
        - ploy
        - rbf
    3. degree: indicates the height of a dimension.
    4. gamma: The larger the value, the more complex the classification boundary can be.
        - auto
        - 0.7
    
    ### Linear model(model 1)
    "C" : 1, 
    "max_iter" : 10000
    ### Non-Linear model (model 2)
    "kernel": 'poly', 
    "degree": 3, 
    "gamma": 'auto',
    "C": 1

    ## performance
    ### 1st time 
    > remove "Cabin" column
    > remove Nan value in "Age" & "Fare" column in test dataset
    #### model 1
    "Correct Rate": 0.6072
    #### model 2 
    "Correct Rate": 0.5589
    ```

2. Random Forest
    ```python
    ## data columns
    "Survived": int
    "Pclass": int
    "Age": int
    "SibSp": int
    "Parch": int
    "Fare": float

    ## model configs
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": 42

    ## performance
    ### 1st time
    "Correct Rate": 0.6363
    ### 2nd time
    > remove "Cabin" column
    > remove Nan value in "Age" & "Fare" column in test dataset
    "Correct Rate": 0.6253
    ```

3. Gradient Boosting
4. XGBoost
    ```python
    ## data columns
    "Survived": int
    "Pclass": int
    "Age": int
    "SibSp": int
    "Parch": int
    "Fare": float
    "Sex": category
    "Cabin": category
    "Embarked": category

    ## model configs
    "objective": "binary:logistic",
    "max_depth": 6,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample": 0.8,
    "n_estimators": 300,
    "enable_categorical": True

    ## performance
    ### 1 st time
    "Correct Rate": 0.8397
    ### 2nd time
    > remove "Cabin" column
    > remove Nan value in "Age" & "Fare" column in test dataset
    "Correct Rate": 0.8368
    ```
5. Neural Network

## 📝 Learning Notes
### SVM
1. Reference Reading
    - [Support Vector Machine](https://ithelp.ithome.com.tw/m/articles/10270447)
2. Two types of SVM
    - Linear SVM
    - Non-Linear SVM: Use **kernel function** to map the data to a high-dimensional space and do seperate. 
        - Polynomial 
        - Radial Basis Function

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
5. Scikit-learn python package is limited to categorical data type, there are some options to solve:
        1. Stay with scikit-learn, but encode the categorical columns.
        2. Some tree libraries natively support categorical features.

6. Different approach testing
    1. Use dataset without categorical data.
        - outcomes: the correct rate decline than XGBoost.
    2. Use dataset with categorical but encoded data.
        - outcomes: We meet `Feature Mismatch` issue, due to encoding process. The unique values in a specific columns aren't same between training and testing 

7. Useful Reading:
    - [Random Forest, Explained: A Visual Guide with Code Examples](https://medium.com/data-science/random-forest-explained-a-visual-guide-with-code-examples-9f736a6e1b3c)

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


