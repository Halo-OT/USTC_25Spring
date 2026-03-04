# Q1:机器学习算法预测TEACHBEHA特征的实验报告

## 摘要



本实验旨在探索和比较不同机器学习算法对TEACHBEHA特征的预测性能。通过对数据集进行预处理、特征选择、模型训练和参数调优，我们发现随机森林和支持向量回归(SVR)模型在该预测任务上表现最佳。实验结果表明，合理的特征选择和参数调优可以显著提升模型性能。

## 1. 数据信息和预处理

### 1.1 数据集概述

本实验使用了教育相关的数据集，包含多种特征变量，目标变量为TEACHBEHA（教师行为）。数据集共包含1048个样本和多个特征列。

### 1.2 数据预处理

数据预处理步骤主要包括：

- **缺失值处理**：
  - 删除目标变量TEACHBEHA缺失的行
  - 对数值型变量使用均值填充
  - 对分类型变量使用众数填充

- **异常值检测**：
  - 使用箱线图和IQR方法识别目标变量中的异常值
  - 由于异常值数量较少且可能包含有价值的信息，我们选择保留这些异常值

- **特征编码**：
  - 对分类变量进行独热编码，去除第一列以避免多重共线性

预处理后，数据集包含了完整的特征集和目标变量，没有缺失值。

```python
# 数据预处理: 处理缺失值和异常值

# 检查并删除常量列
constant_cols = df.columns[df.nunique() == 1]
if len(constant_cols) > 0:
    print(f"删除的常量列: {constant_cols.tolist()}")
    df = df.drop(columns=constant_cols)

# 处理目标变量缺失的行
df_clean = df.dropna(subset=['TEACHBEHA'])

# 检查是否有过多缺失值的列
missing_threshold = 30.0  # 超过30%的缺失值则删除该列
high_missing_cols = info_df[info_df['缺失比例'] > missing_threshold].index.tolist()
if high_missing_cols:
    print(f"删除的高缺失列: {high_missing_cols}")
    df_clean = df_clean.drop(columns=high_missing_cols)

# 处理剩余缺失值
numerical_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df_clean.select_dtypes(exclude=['number']).columns.tolist()

# 数值型特征使用均值填充
for col in numerical_cols:
    if col != 'TEACHBEHA' and df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

# 分类特征使用众数填充
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# 检查处理后的缺失值情况
print(f"\n处理后数据集形状: {df_clean.shape}")
print(f"处理后缺失值总数: {df_clean.isnull().sum().sum()}")
```

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250519161240926.png" alt="image-20250519161240926" style="zoom:50%;" />

## 2. 数据集划分

为了全面评估模型性能，我们采用了70:15:15的比例将数据集划分为训练集、验证集和测试集：

- **训练集**：用于模型训练，占总数据集的70%
- **验证集**：用于模型选择和参数调优，占总数据集的15% 
- **测试集**：用于最终模型评估，占总数据集的15%

划分过程使用了固定的随机种子(42)以确保实验的可重复性：

```python
# 数据集划分 (70% 训练，15% 验证，15% 测试)
# 首先划分出测试集
X_temp, X_test, y_temp, y_test = train_test_split(
    X_encoded, y, test_size=0.15, random_state=42)

# 然后从剩下的数据中划分出训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=42)
```

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250519161421745.png" alt="image-20250519161421745" style="zoom:50%;" />

## 3. 机器学习算法模型

本实验选择了以下六种机器学习算法进行比较：

### 3.1 线性回归 (Linear Regression)

线性回归是一种基础的回归算法，假设目标变量与特征之间存在线性关系。它试图找到一个线性方程，使得预测误差的平方和最小化[1]。

### 3.2 岭回归 (Ridge Regression)

岭回归是线性回归的正则化版本，通过在损失函数中添加L2正则化项来解决多重共线性问题。这种方法在处理特征较多或特征间存在高度相关性的情况下尤为有效[2]。

### 3.3 随机森林回归 (Random Forest Regression)

随机森林是一种集成学习方法，通过构建多棵决策树并取平均值来进行预测。它能够处理高维数据，自动处理特征之间的交互，并提供特征重要性评估[3]。

### 3.4 梯度提升回归 (Gradient Boosting Regression)

梯度提升是一种通过迭代训练一系列弱学习器来构建强学习器的集成方法。每个新的学习器都试图纠正前一个学习器的错误，从而不断提高模型性能[4]。

### 3.5 支持向量回归 (Support Vector Regression, SVR)

SVR是支持向量机算法在回归问题上的应用。它通过构建一个超平面或超平面集，使得尽可能多的数据点位于ε-宽的管道内，同时最小化预测误差[5]。

### 3.6 神经网络回归 (MLP Regressor)

多层感知机(MLP)是一种前馈神经网络，通过多层非线性变换来建模复杂的关系。它能够学习特征之间的非线性交互，但可能需要较多的训练数据[6]。

## 4. 特征选择与处理

基于实验三的相关性分析结果和当前预测任务的需求，我们采用了以下特征处理策略：

### 4.1 特征标准化

使用StandardScaler对所有特征进行标准化处理，使其均值为0，标准差为1：

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

### 4.2 特征选择

采用SelectKBest方法，基于F检验统计量(f_regression)选取与目标变量最相关的10个特征：

```python
selector = SelectKBest(f_regression, k=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)
```

输出：

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250519162011175.png" alt="image-20250519162011175" style="zoom:50%;" />

特征重要性可视化结果：

![image-20250519162101558](/Users/halo/Library/Application Support/typora-user-images/image-20250519162101558.png)

## 5. 主实验：模型训练与评估

### 5.1 评估指标

我们使用以下指标来评估模型性能：
- **均方误差(MSE)**: 预测值与实际值差值的平方的均值
- **均方根误差(RMSE)**: MSE的平方根，与目标变量具有相同单位
- **平均绝对误差(MAE)**: 预测值与实际值差值绝对值的均值
- **决定系数(R²)**: 衡量模型解释因变量变异性的比例，值越接近1表示模型越好

### 5.2 模型训练与验证集评估

训练和评估代码如下：

```py
# 定义评估函数
def evaluate_model(model, X_train, y_train, X_val, y_val):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 在训练集和验证集上预测
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # 计算评估指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    return {
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'val_mse': val_mse,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2
    }
```

```py
# 定义要评估的模型
models = {
    '线性回归': LinearRegression(),
    '岭回归': Ridge(alpha=1.0, random_state=42),
    '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
    '梯度提升': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', gamma='scale'),
    '神经网络': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
}

# 评估所有模型
results = {}

for name, model in models.items():
    print(f"训练和评估 {name}...")
    results[name] = evaluate_model(model, X_train_selected, y_train, X_val_selected, y_val)
```

```py
# 创建结果汇总表
results_df = pd.DataFrame({
    '训练集MSE': [results[model]['train_mse'] for model in models.keys()],
    '训练集RMSE': [results[model]['train_rmse'] for model in models.keys()],
    '训练集MAE': [results[model]['train_mae'] for model in models.keys()],
    '训练集R²': [results[model]['train_r2'] for model in models.keys()],
    '验证集MSE': [results[model]['val_mse'] for model in models.keys()],
    '验证集RMSE': [results[model]['val_rmse'] for model in models.keys()],
    '验证集MAE': [results[model]['val_mae'] for model in models.keys()],
    '验证集R²': [results[model]['val_r2'] for model in models.keys()]
}, index=models.keys())

# 按验证集RMSE排序
results_df = results_df.sort_values('验证集RMSE')

print("模型性能比较 (按验证集RMSE排序):")
display(results_df)
```

所有模型都使用相同的训练数据和特征选择方法进行训练，并在验证集上进行初步评估：

![image-20250519162501666](/Users/halo/Library/Application Support/typora-user-images/image-20250519162501666.png)

可视化结果：

![image-20250519162543954](/Users/halo/Library/Application Support/typora-user-images/image-20250519162543954.png)



基于验证集结果，**随机森林**和**SVR模型**表现最佳，因此选择这两种模型进行进一步的参数调优。

### 5.3 测试集评估调优前的模型性能

在初步评估后，我们在测试集上评估了两个最优模型的性能：

```py
# 在测试集上评估所有模型
test_results = {}

for name, model in models.items():
    # 使用已训练的模型直接预测
    y_test_pred = model.predict(X_test_selected)
    
    # 计算评估指标
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    test_results[name] = {
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }
```

```py
# 创建验证集结果汇总表
val_results_df = pd.DataFrame({
    '验证集MSE': [results[model]['val_mse'] for model in models.keys()],
    '验证集RMSE': [results[model]['val_rmse'] for model in models.keys()],
    '验证集MAE': [results[model]['val_mae'] for model in models.keys()],
    '验证集R²': [results[model]['val_r2'] for model in models.keys()]
}, index=models.keys())

# 按验证集RMSE排序
val_results_df = val_results_df.sort_values('验证集RMSE')

print("\n模型在验证集上的性能比较 (按RMSE排序):")
display(val_results_df)

# 根据验证集性能选择最佳的两个模型
best_models = val_results_df.head(2).index.tolist()
print(f"\n验证集上表现最佳的两个模型: {best_models}")

# 创建测试结果汇总表 (只包含验证集上表现最好的两个模型)
test_results_df = pd.DataFrame({
    '测试集MSE': [test_results[model]['test_mse'] for model in best_models],
    '测试集RMSE': [test_results[model]['test_rmse'] for model in best_models],
    '测试集MAE': [test_results[model]['test_mae'] for model in best_models],
    '测试集R²': [test_results[model]['test_r2'] for model in best_models]
}, index=best_models)

# 按测试集RMSE排序
test_results_df = test_results_df.sort_values('测试集RMSE')

print("\n两个测试集表现最优秀模型在测试集上的性能比较 :")
display(test_results_df)
```

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250519163850227.png" alt="image-20250519163850227" style="zoom:50%;" />

## 6. 参数实验：模型调优

### 6.1 随机森林参数调优

对随机森林模型进行网格搜索，调整以下参数：
- n_estimators: 树的数量
- max_depth: 树的最大深度
- min_samples_split: 分裂内部节点所需的最小样本数
- min_samples_leaf: 叶节点所需的最小样本数

```python
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

调优结果：

![image-20250519164303575](/Users/halo/Library/Application Support/typora-user-images/image-20250519164303575.png)

### 6.2 SVR参数调优

对SVR模型进行网格搜索，调整以下参数：
- kernel: 核函数类型
- C: 正则化参数
- gamma: 核系数
- epsilon: ε-不敏感损失函数中的epsilon值

```python
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'epsilon': [0.1, 0.2, 0.5]
}
```

调优结果：

![image-20250519164413963](/Users/halo/Library/Application Support/typora-user-images/image-20250519164413963.png)

**总体调优代码如下：**

```py
# 创建用于调优的数据集（合并训练集和验证集）
X_train_val = np.vstack((X_train_selected, X_val_selected))
y_train_val = np.concatenate((y_train, y_val))

# 为随机森林和SVR定义参数网格
param_grids = {
    '随机森林': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SVR': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
        'epsilon': [0.1, 0.2, 0.5]
    }
}

# 存储调优前的模型和结果
original_models = {model_name: models[model_name] for model_name in best_models}
original_results = {model_name: test_results[model_name] for model_name in best_models}

# 存储调优后的模型和结果
tuned_models = {}
tuning_results = {}

# 对最佳模型进行调优
for model_name in best_models:
    print(f"为 {model_name} 进行参数调优...")
    param_grid = param_grids[model_name]
    
    # 创建模型副本以避免修改原始模型
    if model_name == '随机森林':
        model = RandomForestRegressor(random_state=42)
    elif model_name == 'SVR':
        model = SVR()
    else:
        model = original_models[model_name]
    
    # 执行网格搜索
    print(f"参数网格大小: {np.prod([len(values) for values in param_grid.values()])} 种组合")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    # 拟合模型
    grid_search.fit(X_train_val, y_train_val)
    
    # 保存最佳模型
    tuned_models[model_name] = grid_search.best_estimator_
    
    # 输出最佳参数和分数
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"交叉验证MSE: {-grid_search.best_score_:.4f}")
    print(f"交叉验证RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
    
    # 在测试集上评估调优后的模型
    y_test_pred = tuned_models[model_name].predict(X_test_selected)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # 保存结果
    tuning_results[model_name] = {
        'best_params': grid_search.best_params_,
        'cv_mse': -grid_search.best_score_,
        'cv_rmse': np.sqrt(-grid_search.best_score_),
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }
    
    print(f"测试集RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}\n")
```

### 6.3 调优效果比较

调优前后模型性能提升比较代码：

```py
# 比较调优前后的性能
tuning_comparison = pd.DataFrame()

for model_name in best_models:
    tuning_comparison[f'{model_name} (调优前)'] = [
        original_results[model_name]['test_rmse'],
        original_results[model_name]['test_r2']
    ]
    tuning_comparison[f'{model_name} (调优后)'] = [
        tuning_results[model_name]['test_rmse'],
        tuning_results[model_name]['test_r2']
    ]

tuning_comparison.index = ['RMSE', 'R²']
print("调优前后性能比较:")
display(tuning_comparison)

# 计算调优带来的提升
for model_name in best_models:
    rmse_improvement = ((original_results[model_name]['test_rmse'] - tuning_results[model_name]['test_rmse']) / 
                       original_results[model_name]['test_rmse'] * 100)
    r2_improvement = ((tuning_results[model_name]['test_r2'] - original_results[model_name]['test_r2']) / 
                     abs(original_results[model_name]['test_r2']) * 100) if original_results[model_name]['test_r2'] != 0 else float('inf')
    
    print(f"{model_name} 调优后的改进:")
    print(f"RMSE改进: {rmse_improvement:.2f}% (降低越多越好)")
    print(f"R²改进: {r2_improvement:.2f}% (增加越多越好)")
```

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250519164642236.png" alt="image-20250519164642236" style="zoom:50%;" />

可视化性能提升：

![image-20250519164745045](/Users/halo/Library/Application Support/typora-user-images/image-20250519164745045.png)



参数调优对两种模型都带来了性能提升，**随机森林**调优的改进不太明显，**SVR模型**在R²指标上的相对改善更为显著。



## 7: 结论分析

### 7.1. 方法的合理性分析

1. **数据预处理的合理性**：
   - 我们处理了目标变量（TEACHBEHA）缺失的行，保留了完整的目标信息，这确保了模型训练的可靠性。
   - 对于其他特征的缺失值，采用了数值型变量均值填充和分类变量众数填充的策略，这是处理缺失值的标准做法，能够保留数据分布的整体特性。
   - 使用独热编码处理分类变量（如STRATUM和PRIVATESCH），有效避免了序数编码可能引入的错误关系假设。

2. **数据集划分的合理性**：
   - 采用70:15:15的比例划分训练集、验证集和测试集，这种划分比例在机器学习实践中很常见，能够确保有足够的数据用于训练，同时保留足够数据用于验证和测试。
   - 使用固定的随机种子（random_state=42）确保了实验的可重复性，便于后续研究或比较不同模型的性能。

3. **特征选择的合理性**：
   - 使用SelectKBest方法选择最重要的10个特征，这一方法基于统计学上的F检验，能够有效识别与目标变量最相关的特征。
   - 特征选择减少了模型复杂度，避免了过拟合，提高了模型的泛化能力，同时也提高了计算效率。

4. **模型选择的合理性**：
   - 选择了多种常见的机器学习算法（线性回归、岭回归、随机森林、梯度提升、SVR、神经网络），覆盖了线性和非线性模型，简单和复杂模型，这种全面的比较确保了我们能找到最适合该数据集的模型。
   - 通过验证集的表现选择表现最佳的两个模型进行进一步的参数调优，这是一种有效的模型选择策略。

5. **评估指标的合理性**：
   - 使用多种评估指标（RMSE、MAE、R²）对模型进行评估，这些指标从不同角度评估了模型的性能，RMSE和MAE衡量预测误差，R²衡量模型解释变异的能力。
   - 在验证集和测试集上分别评估模型，确保了所选模型在新数据上具有良好的泛化能力。

### 7.2. 实验结果分析

1. **模型性能比较**：
   - 随机森林模型在验证集上表现最佳（RMSE=0.75，R²=0.42），这表明数据中可能存在非线性关系，树模型能够更好地捕捉这些关系。
   - SVR模型紧随其后（RMSE=0.78，R²=0.37），确认了非线性模型对该数据集的适用性。
   - 简单的线性模型（线性回归、岭回归）表现相对较差，但差距不是特别大，说明数据中的非线性关系可能不是特别复杂。

2. **特征重要性分析**：
   - STUBEHA（学生行为）是预测TEACHBEHA（教师行为）的最重要特征，这表明学生和教师的行为之间存在强相关性。
   - STAFFSHORT（教职员工短缺）和EDUSHORT（教育资源短缺）也是重要特征，这表明学校资源状况对教师行为有显著影响。
   - 这些发现与教育研究中的常见理论一致，即学校环境、资源和学生特性会影响教师的教学行为。

3. **参数调优结果**：
   - 随机森林模型通过参数调优后，测试集RMSE从0.77降至0.64，R²从0.36提升至0.40，调优效果显著。
   - 最佳参数组合（n_estimators=300, max_depth=20, min_samples_split=2）表明较深但数量适中的决策树组合能够较好地平衡偏差和方差。
   - SVR模型在参数调优后也有所改善，最佳核函数为RBF，这进一步证实了数据中的非线性关系。

### 7.3. 结论依据分析

1. **方法选择的依据**：
   - 基于验证集上的表现选择随机森林和SVR作为最佳模型是合理的，因为这两个模型在验证集上的RMSE和R²均优于其他模型。
   - 数据预处理和特征选择的方法是基于统计学原理和机器学习最佳实践选择的，确保了实验的科学性和可靠性。

2. **结论的可靠性**：
   - 测试集上的结果与验证集一致，都显示随机森林模型表现最佳，这增强了结论的可靠性。
   - 通过参数调优后，模型性能的提升是显著的，这证明了选择的模型和调优方法是有效的。
   - 模型解释了约40%（R²=0.40）的目标变量变异，这在社会科学研究中是一个可接受的水平，表明模型具有一定的预测能力。

3. **结论的局限性**：
   - 即使是最佳模型也只能解释约40%的目标变量变异，这表明还有其他未捕捉到的因素在影响教师行为。
   - 数据集规模有限（约1000个样本），这可能限制了模型的学习能力和泛化性。
   - 该结论可能只适用于类似的教育环境和背景，不一定适用于不同文化或教育系统的情境。

## 参考文献

[1] https://blog.csdn.net/weixin_47999197/article/details/136933104

[2] https://blog.csdn.net/rettbbetter/article/details/129202986

[3] https://blog.csdn.net/goufeng93/article/details/137936811

[4] https://blog.csdn.net/u013172930/article/details/146337779

[5] https://blog.csdn.net/u013172930/article/details/146367659

[6] https://blog.csdn.net/xspyzm/article/details/102840768





# Q2：大语言模型在数据分析中的应用实验报告

## 实验目的
探究大语言模型在数据分析领域的应用能力，包括特征关联性推测和代码生成能力，同时分析不同提示策略对模型输出的影响。

## 实验内容
本次实验分为两个部分：
1. 利用对话式大语言模型进行特征关联性推测，并比较不同交互模式下的输出差异
2. 通过API调用大语言模型生成数据分析代码，并分析不同提示词策略的效果差异

## Q1：对话式大语言模型特征关联性推测

### 实验设计
本实验使用腾讯元宝，通过不同的交互模式要求大语言模型基于数据集特征元信息推测可能存在的特征关联。具体包括三种设置方式：
- 设置A：关闭"深度思考(R1)"，要求模型直接输出结论
- 设置B：关闭"深度思考(R1)"，要求模型逐步思考输出答案
- 设置C：打开"深度思考(R1)"，无额外输出要求

### 实验提示词设计

#### 设置A(直接输出结论)

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250519180843606.png" alt="image-20250519180843606" style="zoom:50%;" />

##### 元宝回答：

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250519180824153.png" alt="image-20250519180824153" style="zoom:50%;" />



#### 设置B(逐步思考)

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250519180952757.png" alt="image-20250519180952757" style="zoom:50%;" />

##### 元宝回答：

>### **逐步分析特征关联性**
>
>#### **1. 分析数据类型和可能的分布**
>
>- 数值型（NUM）：
>  - 连续型（如 `RATCMP1`, `STRATIO`, `SCHSIZE`, `PROATCE`）
>  - 离散型（如 `CLSIZE`, `CREACTIV`, `LANGTEST`）
>- 类别型（CHAR）：
>  - 如 `PRIVATESCH`（公立/私立）、`SCHLTYPE`（学校所有权类型）
>- ID类：
>  - 如 `CNTSCHID`（学校ID）、`STRATUM`（分层ID）
>
>#### **2. 业务逻辑相关性**
>
>**（1）学校资源相关**：
>
>- `SCHSIZE`（学校规模）可能影响 `TOTAT`（教师总数）、`STRATIO`（师生比）
>- `RATCMP1`（计算机数量）可能影响 `RATCMP2`（联网计算机比例）
>- `PRIVATESCH`（学校类型）可能影响 `SCHLTYPE`（所有权）、`PROATCE`（教师认证比例）
>
>**（2）教学质量相关**：
>
>- `PROAT5AB`（本科教师比例）、`PROAT5AM`（硕士教师比例）、`PROAT6`（博士教师比例）可能影响 `EDUSHORT`（教育材料短缺）或 `STAFFSHORT`（师资短缺）
>- `CLSIZE`（班级规模）可能影响 `STRATIO`（师生比）
>
>**（3）学校管理与学生表现**：
>
>- `STUBEHA`（学生行为问题）可能与 `TEACHBEHA`（教师行为问题）相关
>- `SCMCEG`（校长对教师多元文化信念的看法）可能影响 `STUBEHA` 或 `TEACHBEHA`
>
>**（4）权重相关**：
>
>- `W_SCHGRNRABWT`（学校权重）可能与 `W_FSTUWT_SCH_SUM`（学生权重总和）相关
>
>#### **3. 可能的关系类型**
>
>- 线性关系
>
>  （数值型 vs. 数值型）：
>
>  - `SCHSIZE` vs. `TOTAT`（学校规模越大，教师越多）
>  - `STRATIO` vs. `CLSIZE`（师生比可能影响班级规模）
>  - `PROAT5AB` vs. `PROAT5AM`（本科和硕士教师比例可能存在此消彼长）
>
>- 非线性关系
>
>  （如阈值效应）：
>
>  - `RATCMP2`（联网计算机比例）可能在 `RATCMP1`（计算机数量）达到一定数量后趋于稳定
>  - `EDUSHORT`（教育材料短缺）可能在 `SCHSIZE` 过大或过小时更严重
>
>- 类别影响
>
>  （类别型 vs. 数值型）：
>
>  - `PRIVATESCH`（公立/私立）可能影响 `PROATCE`（教师认证比例）
>  - `SCHLTYPE`（所有权类型）可能影响 `SCHSIZE`（学校规模）
>
>#### **4. 可能的因果关系**
>
>- 直接因果：
>  - `SCHSIZE` → `TOTAT`（学校规模导致教师需求增加）
>  - `PRIVATESCH` → `PROATCE`（私立学校可能更倾向于雇佣认证教师）
>- 间接因果：
>  - `PROAT5AM`（硕士教师比例）→ `TEACHBEHA`（教师行为问题）→ `STUBEHA`（学生行为问题）
>  - `RATCMP1`（计算机数量）→ `RATCMP2`（联网计算机比例）→ 影响学生成绩（假设数据中有成绩变量）
>
>#### **5. 总结可能的关联特征对**
>
>| **特征1**       | **特征2**          | **关系类型** | **可能的影响**                 |
>| --------------- | ------------------ | ------------ | ------------------------------ |
>| `SCHSIZE`       | `TOTAT`            | 线性正相关   | 学校规模越大，教师越多         |
>| `STRATIO`       | `CLSIZE`           | 线性相关     | 师生比可能影响班级规模         |
>| `RATCMP1`       | `RATCMP2`          | 非线性相关   | 计算机数量影响联网比例         |
>| `PRIVATESCH`    | `PROATCE`          | 类别影响     | 私立学校教师认证比例更高       |
>| `PROAT5AB`      | `PROAT5AM`         | 线性负相关   | 本科和硕士教师比例可能此消彼长 |
>| `EDUSHORT`      | `STAFFSHORT`       | 线性相关     | 教育材料短缺可能伴随师资短缺   |
>| `STUBEHA`       | `TEACHBEHA`        | 线性相关     | 学生和教师行为问题可能相互影响 |
>| `SCMCEG`        | `TEACHBEHA`        | 非线性相关   | 校长看法可能影响教师行为       |
>| `W_SCHGRNRABWT` | `W_FSTUWT_SCH_SUM` | 线性相关     | 学校权重与学生权重相关         |
>| `SCHLTYPE`      | `SCHSIZE`          | 类别影响     | 所有权类型可能影响学校规模     |
>
>### **最终结论**
>
>以上分析基于特征的业务逻辑和数据类型推测可能的关联性，实际关系需通过统计检验（如相关性分析、回归模型、ANOVA等）验证。

#### 设置C(开启深度思考)

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250519193326172.png" alt="image-20250519193326172" style="zoom:50%;" />

##### 元宝回答：

> ### **特征关联性分析及推测结论**
>
> 基于特征的数据类型、业务逻辑和可能的统计关系，以下是可能相关的特征对及其关系类型：
>
> ------
>
> #### **1. 学校属性相关**
>
> | **特征对**                | **关系类型**       | **推测理由**                        |
> | ------------------------- | ------------------ | ----------------------------------- |
> | `PRIVATESCH` ↔ `SCHLTYPE` | 强类别关联         | 学校类型（公立/私立）决定所有权类型 |
> | `PRIVATESCH` ↔ `STRATIO`  | 数值关联（负相关） | 私立学校可能师生比更低              |
> | `PRIVATESCH` ↔ `RATCMP1`  | 数值关联（正相关） | 私立学校可能拥有更多计算机资源      |
> | `SCHSIZE` ↔ `TOTAT`       | 线性正相关         | 学校规模越大，教师数量越多          |
> | `SCHSIZE` ↔ `STRATIO`     | 非线性（U型）      | 小/大规模学校的师生比可能差异显著   |
>
> ------
>
> #### **2. 教师资源相关**
>
> | **特征对**              | **关系类型** | **推测理由**                       |
> | ----------------------- | ------------ | ---------------------------------- |
> | `TOTAT` ↔ `STRATIO`     | 线性负相关   | 教师数量增加会降低师生比           |
> | `PROATCE` ↔ `PROAT5AB`  | 线性相关     | 认证教师比例可能与学士学位比例相关 |
> | `PROAT5AB` ↔ `PROAT5AM` | 线性负相关   | 学士与硕士教师比例可能此消彼长     |
> | `PROAT5AM` ↔ `PROAT6`   | 线性负相关   | 硕士与博士教师比例可能此消彼长     |
>
> ------
>
> #### **3. 教学质量和资源相关**
>
> | **特征对**                | **关系类型**       | **推测理由**                           |
> | ------------------------- | ------------------ | -------------------------------------- |
> | `RATCMP1` ↔ `RATCMP2`     | 非线性（阈值效应） | 计算机数量多不一定联网比例高           |
> | `EDUSHORT` ↔ `STAFFSHORT` | 线性正相关         | 教育资源短缺常伴随师资短缺             |
> | `STUBEHA` ↔ `TEACHBEHA`   | 线性正相关         | 学生与教师行为问题可能相互影响         |
> | `SCMCEG` ↔ `STUBEHA`      | 非线性相关         | 校长的多元文化观念可能缓解学生行为问题 |
>
> ------
>
> #### **4. 权重相关（统计调整变量）**
>
> | **特征对**                           | **关系类型** | **推测理由**                 |
> | ------------------------------------ | ------------ | ---------------------------- |
> | `W_SCHGRNRABWT` ↔ `W_FSTUWT_SCH_SUM` | 线性正相关   | 学校权重与学生权重总和相关   |
> | `W_FSTUWT_SCH_SUM` ↔ `SENWT`         | 线性相关     | 学生权重总和可能影响国家权重 |
>
> ------
>
> ### **验证方法建议**
>
> 1. 数值型关系：
>    - 使用 **Pearson/Spearman相关系数** 检验线性或单调关系（如 `SCHSIZE` ↔ `TOTAT`）。
>    - 用 **散点图+回归线** 观察非线性趋势（如 `RATCMP1` ↔ `RATCMP2`）。
> 2. 类别型关系：
>    - 使用 **ANOVA** 或 **卡方检验** 验证类别差异（如 `PRIVATESCH` 对 `STRATIO` 的影响）。
> 3. 因果关系：
>    - 构建 **回归模型**（如 `STRATIO ~ SCHSIZE + TOTAT`）控制混杂变量。
>
> 以上推测需结合实际数据验证，部分关系可能受地区、政策等外部因素干扰。

### 实验结果与分析

#### 设置A(直接输出结论)的输出特点
在这种模式下，模型直接给出了可能存在关联的特征对，输出简洁明了，但缺乏推理过程和关联强度的评估。模型主要基于字段名称的语义关系进行推测，例如将"STUBEHA(学生行为问题)"与"TEACHBEHA(教师行为问题)"配对，将"EDUSHORT(教育资源短缺)"与"STAFFSHORT(教职员工短缺)"配对。这种输出缺乏深度分析，可能会遗漏一些非直观但有统计显著性的关联。

#### 设置B(逐步思考)的输出特点
在这种模式下，模型提供了详细的推理过程，按照规定的步骤进行思考。模型首先分析了每个特征的可能数据类型和分布，然后从教育学理论出发探讨了可能的关联关系，接着考虑了线性和非线性关系的可能性，最后给出了综合性结论。这种输出更全面、系统，不仅指出了特征对的关联，还尝试解释了关联的可能机制和理论依据，但分析过程较冗长。

#### 设置C(开启深度思考)的输出特点
在开启深度思考模式下，模型在无额外指导的情况下自动进行了系统化分析。输出内容既有结构化的思考过程，又有简明扼要的结论。相比设置B，深度思考模式下的分析更加整合且深入，模型主动构建了特征之间的逻辑关系网络，并在分析中考虑了潜在的直接关系和间接关系。输出包含了对数据结构的推断、变量间相关性强度、变量间可能关联的多维度分析，并且给出了验证方法建议。

#### 三种设置输出与实验三及T1-S4特征选择的对比
对比三种设置的输出与之前的数据分析结论：

1. **与实验三的相似之处**：
   - 三种设置都识别出了STUBEHA与TEACHBEHA之间的潜在关联，这与实验三中发现的这两个特征间存在显著相关性一致
   - 设置B和C都提到了EDUSHORT与STAFFSHORT的关联，这也与实验三的相关性分析结果吻合
   - 设置C中提出的学校规模(SCHSIZE)与师生比(STRATIO)的关联在实验三中也有所体现

2. **与T1-S4特征选择的相似之处**：
   - 设置B和C都强调了学校特征(如SCHLTYPE、PRIVATESCH)与教育质量指标的关系，这与T1-S4中选择这些特征作为重要预测变量的依据相符
   - 三种设置都识别出了与学生行为和教师行为相关的特征集合，这与T1-S4中对行为类特征的重视一致

3. **差异点**：
   - 设置A由于直接输出结论，缺少了对多变量之间复杂关系的分析，因此与实验三的多变量回归分析结果有较大差异
   - 设置B和C虽然提出了许多可能的关联，但有些并不在实验三的显著结果中，说明模型基于名称语义的推测与实际数据分析之间存在一定差距

总体而言，设置C(开启深度思考)的输出与实际数据分析结果最为接近，提供了更均衡和全面的特征关联分析，既包含了基于先验知识的推测，又融入了对数据结构的思考，更容易捕捉到非显而易见的关联。

## Q2：通过API调用DeepSeek-V3生成数据分析代码

### 实验设计
本实验通过调用DeepSeek-V3的API接口，使用不同的提示策略要求模型生成用于计算特定特征相关系数的Python代码。实验比较了两种提示方式：
- 方式A：直接提问，不提供参考样例
- 方式B：在提问前提供多个相关任务的示例

### 实验实现

#### 数据预处理
首先对实验所需的数据集进行加载和预处理：

```py
# 导入必要的库
import pandas as pd
import numpy as np

# 读取数据集并进行预处理
df = pd.read_csv('/Users/halo/Desktop/数据分析及实践实验/实验5/subdata.csv', index_col=0)

# 处理目标变量缺失的行
df_clean = df.dropna(subset=['TEACHBEHA'])

# 处理剩余缺失值
numerical_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df_clean.select_dtypes(exclude=['number']).columns.tolist()

# 数值型特征使用均值填充
for col in numerical_cols:
    if col != 'TEACHBEHA' and df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

# 分类特征使用众数填充
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
```

#### 方式A: 无样例提示实现

```py
from openai import OpenAI

# 初始化 OpenAI 客户端，使用 DeepSeek 的 API
client = OpenAI(
    api_key="sk-97f35a716186422381889aa649d431f3",
    base_url="https://api.deepseek.com/v1"
)

# 设计输入提示词 - 直接问问题
prompt = """[Question] 已知给定pandas.DataFrame实例df，请编写Python代码求特征STUBEHA与TEACHBEHA、EDUSHORT与STAFFSHORT的相关系数。"""

# 发送请求
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.1,  # 低温度以获得更确定性的输出
    max_tokens=1000
)
```

##### **方式A的输出结果**

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250519200036693.png" alt="image-20250519200036693" style="zoom:50%;" />

大模型给出的代码为：

```py
import pandas as pd

# 假设df是已经存在的DataFrame
# 计算STUBEHA与TEACHBA的相关系数
corr1 = df['STUBEHA'].corr(df['TEACHBEHA'])

# 计算EDUSHORT与STAFFSHORT的相关系数
corr2 = df['EDUSHORT'].corr(df['STAFFSHORT'])

# 打印结果
print(f"STUBEHA与TEACHBEHA的相关系数: {corr1:.4f}")
print(f"EDUSHORT与STAFFSHORT的相关系数: {corr2:.4f}")
```

输出为：

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250519200128201.png" alt="image-20250519200128201" style="zoom:50%;" />

#### **方式B: 带样例提示实现**

```py
# 设计输入提示词 - 提供样例的提示词
prompt_with_examples = """下面是一些使用pandas计算数据统计信息的示例:

[Question] 已知给定pandas.DataFrame实例df，请编写一段Python代码输出列A和列B的平均值。
[Answer] ```print(df['A'].mean(), df['B'].mean())```

[Question] 已知给定pandas.DataFrame实例df，请编写一段Python代码计算列C的标准差。
[Answer] ```print(df['C'].std())```

[Question] 已知给定pandas.DataFrame实例df，请编写一段Python代码按照列D对数据进行排序。
[Answer] ```sorted_df = df.sort_values(by='D')
print(sorted_df)```

[Question] 已知给定pandas.DataFrame实例df，请编写Python代码求特征STUBEHA与TEACHBEHA、EDUSHORT与STAFFSHORT的相关系数。
[Answer]"""

# 发送请求
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": prompt_with_examples}
    ],
    temperature=0.1,
    max_tokens=1000
)
```

##### **方式B的输出结果**

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250519200156826.png" alt="image-20250519200156826" style="zoom:50%;" />

大模型给出的代码为：

```py
# 计算STUBEHA与TEACHBEHA的相关系数
corr1 = df['STUBEHA'].corr(df['TEACHBEHA'])
# 计算EDUSHORT与STAFFSHORT的相关系数
corr2 = df['EDUSHORT'].corr(df['STAFFSHORT'])

print(f"STUBEHA与TEACHBEHA的相关系数: {corr1}")
print(f"EDUSHORT与STAFFSHORT的相关系数: {corr2}")
```

输出为：

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250519200241076.png" alt="image-20250519200241076" style="zoom:50%;" />

### 实验结果分析

#### 输出内容对比

1. **代码结构**：
   - 无样例提示(方式A)生成的代码包含导入语句、详细注释以及格式化输出
   - 带样例提示(方式B)生成的代码更加简洁，没有冗余的导入语句，注释更少，与样例风格一致

2. **代码格式**：
   - 无样例提示的代码会主动格式化输出结果(`{corr1:.4f}`)控制小数位数
   - 带样例提示的代码没有进行格式化处理，更简洁直接(`{corr1}`)

3. **注释风格**：
   - 无样例提示的代码包含更详细的注释，例如"假设df是已经存在的DataFrame"
   - 带样例提示的代码注释较少，更接近样例中的风格

4. **代码执行有效性**：
   - 两种方法生成的代码都能在数据预处理完成的情况下正常执行并得到正确结果
   - 但方式A生成的代码有一个小错误，将"TEACHBEHA"写成了"TEACHBA"，表明没有样例时容易出现拼写错误

#### 提示工程效果分析

1. **样例的引导作用**：
   - 提供样例明显引导了模型生成符合特定风格的代码
   - 样例不仅影响代码的格式，还影响了注释风格和输出格式
4. **代码简洁性**：
   - 带样例提示生成的代码更加简洁，去除了不必要的元素
   - 这表明样例可以有效控制输出的冗余程度

#### 结论：

针对特定任务设计合适的提示词和样例，能有效引导模型生成更符合预期的代码输出

