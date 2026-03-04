# PISA数据集频繁项集挖掘与关联规则分析实验报告



## 1. 实验目的

本实验基于PISA 2018学校问卷数据集，通过频繁项集挖掘和关联规则分析，探究学校数字资源配置的内在关联模式。实验主要目标是：
1. 对原始问卷数据进行预处理，构建适合关联规则挖掘的数据集
2. 使用Apriori算法挖掘频繁项集，比较不同支持度阈值下的结果
3. 通过数据二值化探索学校数字资源配置的关联规则
4. 分析挖掘结果，总结学校数字资源配置的关键特征和模式



## 2. 数据描述

本实验使用了PISA 2018学校问卷数据中与学校数字资源相关的5个特征：

- SC155Q01HA：连接到互联网的数字设备数量是否充足
- SC155Q02HA：学校的互联网带宽或速度是否充足
- SC155Q03HA：用于教学的数字设备数量是否充足
- SC155Q04HA：学校数字设备的计算能力是否足够强大
- SC155Q05HA：是否有充足的适当软件可用

这些特征的原始取值为1-4，分别表示回答者对相关问题的认同程度：
1. 强烈不同意(Strongly disagree)
2. 不同意(Disagree)
3. 同意(Agree)
4. 强烈同意(Strongly agree)



## 3. 方法与实现

### 3.1 数据预处理

首先对原始数据进行预处理，主要步骤包括：

1. 特征选取与重命名：选取5个关键特征并进行简化命名
   ```python
   feature_mapping = {
       'SC155Q01HA': 'internet_devices',
       'SC155Q02HA': 'internet_bandwidth',
       'SC155Q03HA': 'instruction_devices',
       'SC155Q04HA': 'computing_capacity',
       'SC155Q05HA': 'adequate_software'
   }
   ```

2. 缺失值处理：删除包含缺失值的样本
   ```python
   df.dropna(inplace=True)
   ```

3. 项集索引构建：为频繁项集挖掘创建特征值对应的唯一索引
   ```python
   ind2val = {}  # 索引到特征值的映射
   val2ind = {}  # 特征值到索引的映射
   ```



下面回答Q1和Q2并粘贴全部代码

#### Q1. 特征含义与简化
选取的5个特征实际上与学校的数字设备及互联网资源有关，具体含义如下：

- SC155Q01HA:   连接到互联网的数字设备数量是否充足
- SC155Q02HA:   学校的互联网带宽或速度是否充足
- SC155Q03HA: 用于教学的数字设备数量是否充足
- SC155Q04HA: 学校数字设备的计算能力是否足够强大
- SC155Q05HA: 是否有充足的适当软件可用


这些特征的取值范围为1-4，表示认同程度： 1: 强烈不同意 (Strongly disagree) 2: 不同意 (Disagree) 3: 同意 (Agree) 4: 强烈同意 (Strongly agree)

下面是数据预处理代码：

```py
import pandas as pd
import numpy as np

# 读取数据集
data = pd.read_csv('data.csv')

# 简化特征名称映射
feature_mapping = {
    'SC155Q01HA': 'internet_devices',
    'SC155Q02HA': 'internet_bandwidth',
    'SC155Q03HA': 'instruction_devices',
    'SC155Q04HA': 'computing_capacity',
    'SC155Q05HA': 'adequate_software'
}

# 选取需要的5个特征
selected_features = ['SC155Q01HA', 'SC155Q02HA', 'SC155Q03HA', 'SC155Q04HA', 'SC155Q05HA']
df = data[selected_features].copy()

# 重命名列
df.rename(columns=feature_mapping, inplace=True)

# 删除存在缺失值的行
df.dropna(inplace=True)

print("原始特征数量:", len(data))
print("删除缺失值后的特征数量:", len(df))
print("\n特征含义:")
for orig, simple in feature_mapping.items():
    print(f"{orig} -> {simple}")

print("\n各特征取值范围:")
for col in df.columns:
    print(f"{col}: {sorted(df[col].unique())}")
```

**输出**：<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250503121618041.png" alt="image-20250503121618041" style="zoom:50%;" />

#### Q2 构建项集索引

```py
# 构建项集索引
ind2val = {}  # 索引到特征值的映射
val2ind = {}  # 特征值到索引的映射
index = 0

# 为每个特征-值对创建一个唯一索引
for col in df.columns:
    for val in sorted(df[col].unique()):
        key = f'{col}={val}'
        ind2val[index] = key
        val2ind[key] = index
        index += 1

print("\n项集索引字典:")
for idx, val in sorted(ind2val.items()):
    print(f"{idx}: '{val}'")

# 创建一个新的数据框来存储转换后的数据
df_transformed = pd.DataFrame(index=df.index)
items_list = []

# 对每行进行转换，将特征值替换为对应的索引值
for idx, row in df.iterrows():
    row_items = []
    for col in df.columns:
        if not pd.isna(row[col]):  # 确保不处理缺失值
            item_key = f'{col}={row[col]}'
            if item_key in val2ind:
                row_items.append(val2ind[item_key])
    items_list.append(row_items)
    
df_transformed = pd.DataFrame({'items': items_list}, index=df.index)

print("\n转换后的数据前5行:")
print(df_transformed.head())
```

**输出**：

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250503121754027.png" alt="image-20250503121754027" style="zoom:50%;" />

### 3.2 频繁项集挖掘

下面是对Q1～3的回答和代码实现：

#### Q1. Apriori算法实现

首先，我们使用Apriori算法进行频繁项集挖掘，分别使用0.25和0.5的最小支持度阈值：

```py
from collections import defaultdict
from itertools import combinations  

def apriori_algorithm(transactions, min_support):
    """
    严格按照伪代码实现的Apriori算法
    
    参数:
    transactions: 事务列表，每个事务是项的列表
    min_support: 最小支持度阈值
    
    返回:
    所有频繁项集的集合
    """
    # 总事务数
    n_transactions = len(transactions)
    
    # 步骤1: k = 1
    k = 1
    
    # 获取所有可能的1-项集
    all_items = set()
    for transaction in transactions:
        for item in transaction:
            all_items.add(frozenset([item]))
    
    # 计算每个1-项集的支持度
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in all_items:
            if item.issubset(set(transaction)):
                item_counts[item] += 1
    
    # 步骤2: 找出频繁1-项集 F₁
    F = {1: {item: count for item, count in item_counts.items() 
             if count >= min_support * n_transactions}}
    
    # 步骤3-13: 重复直到找不到更多频繁项集
    while F.get(k) and len(F[k]) > 0:
        k += 1  # 步骤4: k = k+1
        
        # 步骤5: 生成候选k-项集 C_k = apriori-gen(F_{k-1})
        C_k = apriori_gen(F[k-1])
        
        # 步骤6-11: 计算候选项集的支持度
        candidate_counts = defaultdict(int)
        for transaction in transactions:
            # 步骤7: 识别属于t的所有候选
            C_t = subset(C_k, transaction)
            
            # 步骤8-10: 更新支持度计数
            for candidate in C_t:
                candidate_counts[candidate] += 1
        
        # 步骤12: 提取频繁k-项集
        F[k] = {item: count for item, count in candidate_counts.items() 
                if count >= min_support * n_transactions}
    
    # 步骤14: 返回所有频繁项集的并集
    result = {}
    for k_value, frequent_items in F.items():
        if frequent_items:  # 确保不为空
            result[k_value] = [(item, count/n_transactions) for item, count in frequent_items.items()]
    
    return result

def apriori_gen(F_prev):
    """
    实现apriori-gen函数，生成候选项集
    """
    candidates = set()
    prev_items = list(F_prev.keys())
    
    for i in range(len(prev_items)):
        for j in range(i+1, len(prev_items)):
            # 对(k-1)-项集两两合并
            item1 = prev_items[i]
            item2 = prev_items[j]
            
            # 如果前k-2个项相同，合并这两个集合
            if len(item1.union(item2)) == len(item1) + 1:
                new_candidate = item1.union(item2)
                
                # 剪枝：检查所有(k-1)子集是否都是频繁的
                is_valid = True
                for subset in [frozenset(s) for s in combinations(new_candidate, len(item1))]:
                    if subset not in F_prev:
                        is_valid = False
                        break
                
                if is_valid:
                    candidates.add(new_candidate)
    
    return candidates

def subset(C_k, transaction):
    """
    实现subset函数，找出事务t中所有可能的候选项集
    """
    transaction_set = set(transaction)
    return [c for c in C_k if c.issubset(transaction_set)]


# 创建事务列表（使用前面已转换的数据）
transactions = df_transformed['items'].tolist()

# 使用最小支持度0.25挖掘频繁项集
frequent_itemsets_025 = apriori_algorithm(transactions, 0.25)

# 显示结果
print("最小支持度为0.25的频繁项集:")
total_025 = 0
for k, itemsets in frequent_itemsets_025.items():
    print(f"频繁{k}-项集数量: {len(itemsets)}")
    total_025 += len(itemsets)
    if k <= 2:  # 为了简洁，只显示部分结果
        for itemset, support in itemsets[:5]:  # 显示前5个
            print(f"  项集: {[ind2val[i] for i in itemset]}, 支持度: {support:.4f}")
    else:
        print(f"  (显示部分结果...共{len(itemsets)}个{k}-项集)")
print(f"频繁项集总数: {total_025}")

# 使用最小支持度0.5挖掘频繁项集
frequent_itemsets_05 = apriori_algorithm(transactions, 0.5)

# 显示结果
print("\n最小支持度为0.5的频繁项集:")
total_05 = 0
for k, itemsets in frequent_itemsets_05.items():
    print(f"频繁{k}-项集数量: {len(itemsets)}")
    total_05 += len(itemsets)
    for itemset, support in itemsets[:5]:  # 显示前5个
        print(f"  项集: {[ind2val[i] for i in itemset]}, 支持度: {support:.4f}")
print(f"频繁项集总数: {total_05}")
```

**输出**：<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250503122001382.png" alt="image-20250503122001382" style="zoom:50%;" />

#### Q2. 数据二值化与频繁项集挖掘

现在我们将数据进行二值化处理，将原始值1和2改为0，将3和4改为1：

```py
# 数据二值化处理
df_binary = df.copy()
for col in df_binary.columns:
    # 将1和2的值修改为0（不同意）
    df_binary.loc[df_binary[col].isin([1, 2]), col] = 0
    # 将3和4的值修改为1（同意）
    df_binary.loc[df_binary[col].isin([3, 4]), col] = 1

print("二值化后的数据前5行:")
print(df_binary.head())

# 重新构建项集索引
binary_ind2val = {}
binary_val2ind = {}
index = 0

for col in df_binary.columns:
    for val in sorted(df_binary[col].unique()):
        key = f'{col}={val}'
        binary_ind2val[index] = key
        binary_val2ind[key] = index
        index += 1

print("\n二值化后的项集索引字典:")
for idx, val in sorted(binary_ind2val.items()):
    print(f"{idx}: '{val}'")

# 创建二值化后的事务列表
binary_transactions = []
for idx, row in df_binary.iterrows():
    row_items = []
    for col in df_binary.columns:
        if not pd.isna(row[col]):
            item_key = f'{col}={row[col]}'
            if item_key in binary_val2ind:
                row_items.append(binary_val2ind[item_key])
    binary_transactions.append(row_items)

# 使用最小支持度0.5挖掘二值化数据的频繁项集
binary_frequent_itemsets = apriori_algorithm(binary_transactions, 0.5)

# 显示结果
print("\n二值化后，最小支持度为0.5的频繁项集:")
total_binary = 0
for k, itemsets in binary_frequent_itemsets.items():
    print(f"频繁{k}-项集数量: {len(itemsets)}")
    total_binary += len(itemsets)
    for itemset, support in itemsets[:5]:  # 每个k只显示前5个
        print(f"  项集: {[binary_ind2val[i] for i in itemset]}, 支持度: {support:.4f}")
print(f"频繁项集总数: {total_binary}")
```

**输出**：

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250506231257883.png" alt="image-20250506231257883" style="zoom:50%;" />

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250506231235214.png" alt="image-20250506231235214" style="zoom:50%;" />

#### Q3.分析发现

1. 支持度阈值影响：
    - 当支持度从0.25提高到0.5时，频繁项集的数量显著减少。这是因为较高的支持度要求项集在更多的事务中出现。
    - 支持度0.25允许相对少见的项集被发现，而支持度0.5则只保留了最常见的项集。
2. 数据二值化的影响：

    - 二值化后（支持度0.5），频繁项集的数量明显增加，这是因为：
        - 数据粒度从4个可能值（1-4）减少到2个可能值（0-1）
        - 原来分散在多个值上的支持度被合并，从而增加了每个项的频率
        - 特别是将"同意"（3和4）合并为1，"不同意"（1和2）合并为0，使得规律更加突出
3. 从特征定义角度分析：

    - 原始数据中各特征有4个取值（强烈不同意、不同意、同意、强烈同意），这种细粒度区分虽然提供了更多信息，但导致支持度分散
    - 二值化后只关注"同意"与"不同意"的区别，忽略了程度上的差异，但更能发现学校数字资源整体状况的模式
    - 从问卷内容看，这5个特征都与学校数字设备和网络资源有关，它们之间可能存在相关性（如网络带宽与设备数量往往相关）
4. 实际含义分析：

    - 通过二值化，我们可能发现多个特征的关联性更强，例如，很多学校可能在多个维度上都表现为"充足"或"不充足"
    - 学校的数字资源往往是系统性投入的结果，如果一个学校有足够的设备，往往也会配备相应的软件和网络
    - 这种关联性在二值化后更容易被发现，导致频繁项集数量增加

### 3.3 关联规则提取与分析

下面是对Q1 Q2的回答和代码实现：

```py
def generate_rules_for_target(frequent_itemsets, min_confidence=0.8, target_item=1):
    """
    生成形如 X -> {target_item} 的关联规则
    
    参数:
    frequent_itemsets: Apriori算法生成的频繁项集
    min_confidence: 最小置信度阈值
    target_item: 目标后件项，这里是索引1（internet_devices=1.0）
    
    返回:
    关联规则列表，每条规则格式为(antecedent, consequent, confidence, lift)
    """
    rules = []
    
    # 存储所有频繁项集的支持度，用于后续计算
    all_itemsets_support = {}
    for k, itemsets in frequent_itemsets.items():
        for itemset, support in itemsets:
            all_itemsets_support[frozenset(itemset)] = support
    
    # 目标后件项的支持度
    target_consequent = frozenset([target_item])
    target_support = all_itemsets_support.get(target_consequent, 0)
    
    if target_support == 0:
        print(f"警告：目标项 {target_item} 不是一个频繁1-项集")
        return rules
    
    # 生成关联规则
    for k, itemsets in frequent_itemsets.items():
        # 至少需要2-项集才能生成规则
        if k < 2:
            continue
            
        for itemset, support in itemsets:
            # 检查该项集是否包含目标后件项
            if target_item in itemset:
                # 前件X (去除目标后件项的剩余项集)
                antecedent = frozenset([i for i in itemset if i != target_item])
                
                # 计算置信度：support(X∪Y) / support(X)
                antecedent_support = all_itemsets_support.get(antecedent, 0)
                if antecedent_support > 0:  # 避免除零错误
                    confidence = support / antecedent_support
                    
                    # 计算提升度：support(X∪Y) / (support(X) * support(Y))
                    lift = support / (antecedent_support * target_support)
                    
                    # 检查是否满足最小置信度要求
                    if confidence >= min_confidence:
                        rules.append((antecedent, target_consequent, confidence, lift))
    
    return rules

# 使用二值化后的频繁项集计算关联规则，目标后件项为'internet_devices=1.0'(索引1)
target_rules = generate_rules_for_target(binary_frequent_itemsets, min_confidence=0.8, target_item=1)

# 显示提取的关联规则
print(f"满足条件的关联规则数量: {len(target_rules)}")
print("\n关联规则 (形如 X -> {internet_devices=1.0}):")
print("前件 -> 后件 [置信度, 提升度]")

# 对关联规则进行排序（先按置信度，再按提升度）
sorted_rules = sorted(target_rules, key=lambda x: (-x[2], -x[3]))

# 显示规则
for antecedent, consequent, confidence, lift in sorted_rules:
    # 将索引转换为更可读的形式
    antecedent_labels = [binary_ind2val[i] for i in antecedent]
    consequent_label = binary_ind2val[list(consequent)[0]]
    print(f"{antecedent_labels} -> {consequent_label} [置信度={confidence:.4f}, 提升度={lift:.4f}]")

# 分析生成的关联规则
print("\n关联规则分析:")
if target_rules:
    avg_confidence = sum(rule[2] for rule in target_rules) / len(target_rules)
    avg_lift = sum(rule[3] for rule in target_rules) / len(target_rules)
    print(f"平均置信度: {avg_confidence:.4f}")
    print(f"平均提升度: {avg_lift:.4f}")
    
    # 找出最强的关联规则（置信度和提升度综合最高）
    strongest_rule = max(target_rules, key=lambda x: x[2] * x[3])
    antecedent_labels = [binary_ind2val[i] for i in strongest_rule[0]]
    print(f"最强关联规则: {antecedent_labels} -> internet_devices=1.0 [置信度={strongest_rule[2]:.4f}, 提升度={strongest_rule[3]:.4f}]")
else:
    print("没有找到满足条件的关联规则")
```

**输出**：

<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250506230930904.png" alt="image-20250506230930904" style="zoom:50%;" />

#### Q2. 关联规则分析与总结



从输出结果看，我们成功掘出了4条形如 X -> `internet_devices=1.0` 的关联规则，每条规则都揭示了学校数字资源配置的内在关系：

1. **教学设备与互联网设备的强关联**：
   - 规则：`instruction_devices=1.0` -> `internet_devices=1.0` [置信度=0.9278, 提升度=1.4841]
   - 解读：当学校拥有充足的教学设备时，有92.78%的概率也拥有充足的互联网设备
   - 这是最强的关联规则，表明教学设备充足与互联网设备充足几乎形成必然关系

2. **计算能力与互联网设备的关联**：
   - 规则：`computing_capacity=1.0` -> `internet_devices=1.0` [置信度=0.8392, 提升度=1.3423]
   - 解读：当学校的数字设备计算能力充足时，有83.92%的概率互联网设备也充足
   - 提示计算能力和联网能力通常一起提升

3. **互联网带宽与设备数量的协同**：
   - 规则：`internet_bandwidth=1.0` -> `internet_devices=1.0` [置信度=0.8263, 提升度=1.3216]
   - 解读：当学校拥有充足的互联网带宽时，有82.63%的概率也拥有充足的互联网设备
   - 表明学校通常会同步提升网络基础设施的质与量

4. **软件资源与互联网设备的关系**：
   - 规则：`adequate_software=1.0` -> `internet_devices=1.0` [置信度=0.8101, 提升度=1.2957]
   - 解读：当学校拥有充足的软件资源时，有81.01%的概率也拥有充足的互联网设备
   - 显示软硬件资源配置的同步性

#### 技术指标分析

1. **置信度**：
   - 所有规则的置信度均超过0.8（最低0.81），平均置信度达0.8509
   - 这表明这些规则具有较高的预测能力，前提条件满足时，结论成立的概率很高

2. **提升度**：
   - 所有规则的提升度均大于1（范围1.29~1.48），平均提升度为1.3609
   - 提升度>1表明这些关联不是随机现象，而是前件对后件有积极的促进作用
   - 例如，教学设备充足使得学校拥有充足互联网设备的可能性比随机情况高出48.41%

#### 教育资源配置的启示

1. **资源协同配置模式**：
   - 各项数字资源之间存在紧密关联，学校倾向于整体提升数字资源水平
   - 很少出现某一资源特别突出而其他资源不足的情况

2. **教学设备的核心地位**：
   - 教学设备充足与互联网设备充足的关联最强，置信度达92.78%
   - 这可能表明学校首先关注的是教学设备，其他资源配置围绕教学需求展开

3. **数字鸿沟现象**：
   - 规则的高置信度表明，一旦学校在某一数字资源维度上表现良好，其他维度也很可能良好
   - 反之亦然，可能存在一些学校各方面资源都不足的情况
   - 这暗示了学校间存在数字资源配置的"两极分化"现象

4. **资源规划策略**：
   - 从关联规则看，学校数字资源建设应当采取整体规划的方式
   - 重点提升教学设备配置可能带动互联网设备等其他资源的改善
   - 政策制定者应关注资源协同发展，避免孤立投入某一单项资源

这些关联规则不仅揭示了学校数字资源配置的内在规律，也为教育资源投入提供了数据支持，有助于更科学地规划学校数字化建设路径。