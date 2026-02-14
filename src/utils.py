# src/utils.py（修改 fillna 部分）
def preprocess_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    full = pd.concat([train, test], sort=False)
    
    # 提取 Title
    full['Title'] = full['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    full['Title'] = full['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    full['Title'] = full['Title'].replace('Mlle', 'Miss')
    full['Title'] = full['Title'].replace('Ms', 'Miss')
    full['Title'] = full['Title'].replace('Mme', 'Mrs')
    
    # 更健壮的 Age 填充：先用 Title 分组，再用全局中位数兜底
    full['Age'] = full.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    full['Age'].fillna(full['Age'].median(), inplace=True)  # 兜底
    
    # Fare 填充
    full['Fare'].fillna(full['Fare'].median(), inplace=True)
    
    # Embarked 填充
    full['Embarked'].fillna(full['Embarked'].mode()[0], inplace=True)
    
    # 创建新特征
    full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
    full['IsAlone'] = (full['FamilySize'] == 1).astype(int)
    
    # 选择特征
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
    full = pd.get_dummies(full[features], columns=['Sex', 'Embarked'], drop_first=True)
    
    # 拆分
    train_clean = full.iloc[:len(train)]
    test_clean = full.iloc[len(train):]
    
    return train_clean.values, test_clean.values, train['Survived'].values, test['PassengerId']