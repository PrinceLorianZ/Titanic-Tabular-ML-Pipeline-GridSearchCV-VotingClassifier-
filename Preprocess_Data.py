import numpy as np
import pandas as pd

#导入python的机器学习算法包 scikit-learn
from sklearn import metrics
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import re

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
#忽略告警提示
import warnings
warnings.filterwarnings("ignore")


# 定义一个函数，从姓名中提取头衔
def get_title(name):
    title_search = re.search(', ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return np.nan


def load_and_combine(path="./data/titanic_data/train.csv"):
    train_src = pd.read_csv(path)
    train_src['Data_type'] = 'train'
    pred_src = pd.read_csv(path)
    pred_src['Data_type'] = 'pred'
    combine = pd.concat([train_src, pred_src])

    ## 对年龄缺失值进行均值填充
    age_mean = round(train_src['Age'].mean())
    combine['Age'].fillna(age_mean, inplace=True)

    # Fare 船票价格 出现空值，用训练集中的平均值替换
    combine['Fare'].fillna(round(train_src['Fare'].mean()), inplace=True)

    # 构建一个Title的新变量
    combine['Title'] = combine['Name'].apply(get_title)
    # 查看title取值
    combine['Title'].value_counts(dropna=False)

    # 相似头衔归为一组，出现次数比较少的,且不能合并在主流数据中的归位一组'Rare'
    combine['Title'] = combine['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    combine['Title'] = combine['Title'].replace('Mlle', 'Miss')
    combine['Title'] = combine['Title'].replace('Ms', 'Miss')
    combine['Title'] = combine['Title'].replace('Mme', 'Mrs')

    # 再次查看title取值
    combine['Title'].value_counts(dropna=False)

    combine['FamilySize'] = combine['SibSp'] + combine['Parch'] + 1
    combine['FamilySize'].value_counts()

    combine['is_mother'] = 0
    combine.loc[(combine['Sex'] == 'female') & (combine['Parch'] > 0) & (combine['Age'] > 20), 'is_mother'] = 1
    combine['is_mother'].value_counts()

    sex_onehot = pd.get_dummies(combine['Sex'], drop_first=False, prefix='onehot')

    # 连续型变量离散化为分类型变量再转为标志型变量（例如年龄分段后，变成Age_0_16_flag…）
    combine['Age_group'] = np.nan
    combine.loc[combine['Age'] <= 16, 'Age_group'] = 'Age_0_16'
    combine.loc[(combine['Age'] > 16) & (combine['Age'] <= 32), 'Age_group'] = 'Age_16_32'
    combine.loc[(combine['Age'] > 32) & (combine['Age'] <= 48), 'Age_group'] = 'Age_32_48'
    combine.loc[(combine['Age'] > 48) & (combine['Age'] <= 64), 'Age_group'] = 'Age_48_64'
    combine.loc[combine['Age'] > 64, 'Age_group'] = 'Age_64_'

    age_group_onehot = pd.get_dummies(combine['Age_group'], drop_first=False, prefix='onehot')

    title_onehot = pd.get_dummies(combine['Title'], drop_first=False, prefix='onehot')

    pclass_onehot = pd.get_dummies(combine['Pclass'], drop_first=False, prefix='onehot_pclass')

    combine = pd.concat([combine, sex_onehot, age_group_onehot, title_onehot, pclass_onehot], axis=1)

    train_X = combine.loc[
        combine['Data_type'] == 'train',
        ['SibSp', 'Parch', 'Fare',
         'FamilySize', 'is_mother',
         'onehot_male', 'onehot_female',
         'onehot_Age_0_16', 'onehot_Age_16_32', 'onehot_Age_32_48',
         'onehot_Age_48_64', 'onehot_Age_64_',
         'onehot_Master', 'onehot_Miss', 'onehot_Mr', 'onehot_Mrs', 'onehot_Rare',
         'onehot_pclass_1', 'onehot_pclass_2', 'onehot_pclass_3']
    ]

    train_y = combine.loc[combine['Data_type'] == 'train', 'Survived']

    X_train, X_test, y_train, y_test = model_selection.train_test_split(train_X, train_y, test_size=0.3,
                                                                        random_state=42)
    return X_train, X_test, y_train, y_test


