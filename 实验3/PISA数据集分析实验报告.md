

# PISA数据集分析实验报告

## 摘要

本实验基于国际学生评估项目(PISA)数据集，对学校环境中的学生行为(STUBEHA)、教师行为(TEACHBEHA)以及教育资源短缺(EDUSHORT)和教职员工短缺(STAFFSHORT)等特征进行了全面分析。通过数据预处理、描述性统计、可视化分析、分布检验、参数估计和假设检验等方法，探究了各特征之间的关系及其分布特性，为理解学校教育环境因素提供了数据支持。

## 1. 实验目的

1. 使用Pandas对数据进行初步处理，包括处理缺失值和冗余列
2. 运用Matplotlib和Numpy实现数据可视化，分析特征之间的关系
3. 进行分布检验，验证特征是否服从正态分布
4. 基于正态分布假设进行参数估计
5. 对特征之间的差异进行假设检验



## 2. 数据描述

本实验使用的数据集来自PISA调查，包含21903行和197列的学校数据。主要关注的特征包括：

- **STUBEHA (学生行为)**: 衡量学校中的学生行为问题或纪律情况
- **TEACHBEHA (教师行为)**: 评估教师的行为、表现或教学态度
- **EDUSHORT (教育资源短缺)**: 衡量学校教育资源的短缺程度
- **STAFFSHORT (教职员工短缺)**: 衡量学校教职员工的短缺程度
- **PRIVATESCH (学校类型)**: 区分公立和私立学校



## 3. 实验方法与实现

### 3.1 数据预处理

使用Pandas进行数据预处理，主要完成以下任务：

1. **Q1 读取数据并统计基本信息**

   代码实现：

   ```py
   import pandas as pd
   df = pd.read_csv('data.csv', index_col=0)
   print(df.head(10))
   print(f"数据集有{df.shape[0]}行和{df.shape[1]}列") # 索引不再当作列数， 表头不计入行数
   ```

   输出：

   <img src="/Users/halo/Library/Application Support/typora-user-images/image-20250403173512230.png" alt="image-20250403173512230" style="zoom:50%;" />

   

2. **Q2 处理缺失值，识别缺失值最多的列**

   代码实现：

   ```py
   df.isnull().sum() # 统计每一列缺失值的个数
   df.drop(df.columns[-1], axis=1, inplace=True) # 删除最后一列
   
   # 展示哪一列的缺失值最多
   print("缺失值最多的列:", df.isnull().sum().idxmax())
   # 找到哪些列缺失值为0
   print("无缺失值的列:", df.columns[df.isnull().sum()==0].tolist())
   ```

   输出：

   <img src="/Users/halo/Library/Application Support/typora-user-images/image-20250403173637724.png" alt="image-20250403173637724" style="zoom:50%;" />

   

3. **Q3 查找并删除冗余列（唯一值列）**

   代码实现：

   ```py
   # 统计每一列的唯一值个数
   constant_cols = df.columns[df.nunique() == 1] 
   
   for col in constant_cols:
       print(f"{col}: {df[col].iloc[0]}") # 统计唯一值个数为1的列
   
   df.drop(columns=constant_cols, inplace=True) # 删除唯一值个数为1的列
   ```

   输出：

   <img src="/Users/halo/Library/Application Support/typora-user-images/image-20250403173806073.png" alt="image-20250403173806073" style="zoom:50%;" />

   对这两列含义的解释：

   - **CYC：**

     在PISA（国际学生能力评估项目）数据中是指**评估周期代码**，它由以下部分组成：

     - 前2位数字：代表评估年份的最后两位数字
     - 后2个字符：代表评估类型

   - **ADMINMODE：**

     PISA数据中的一个变量，它表示**调查或测试的管理方式**，即学生完成评估或问卷的形式。

     

4. **Q4 对分类特征进行归并处理**

   代码实现：

   ```py
   df['PRIVATESCH'] = df['PRIVATESCH'].str.upper() # 将PRIVATESCH列统一转换为大写
   print(df['PRIVATESCH'].value_counts())          # 统计PRIVATESCH列的值
   ```

   输出：

   <img src="/Users/halo/Library/Application Support/typora-user-images/image-20250403175547624.png" alt="image-20250403175547624" style="zoom:50%;" />

   

5. **Q5 统计特征描述性统计和相关性分析**

   代码实现：

   ```py
   # 检查列名是否存在
   columns_to_select = ['STUBEHA', 'TEACHBEHA', 'EDUSHORT', 'STAFFSHORT'] # 选择的列名（实验文档中， 好像打成 TEACHBA 了）
   missing_columns = [col for col in columns_to_select if col not in df.columns]
   if missing_columns:
       print(f"以下列不存在: {missing_columns}")
   else:
       # 确保列是数值类型
       selected = df[columns_to_select].select_dtypes(include=['number'])
       print(selected.describe())  # 统计描述
       print(selected.corr(method='pearson'))      # Pearson 相关性分析
   ```

​	输出：

​	<img src="/Users/halo/Library/Application Support/typora-user-images/image-20250403175821694.png" alt="image-20250403175821694" style="zoom:50%;" />



6. **Q6 对特征 STUBEHA 与 TEACHBEHA 之间、EDUSHORT 与 STAFFSHORT 的相关系数较高的解释：**

   - 列名含义解释：
     - STUBEHA       ： Student Behavior (学生行为)
         - 可能衡量学校中的学生行为问题或纪律情况
         - 负值通常表示更多问题行为，正值表示更好的学生行为/纪律
     - TEACHBEHA     ：Teacher Behavior (教师行为)
         - 评估教师的行为、表现或教学态度
         - 负值可能表示教师行为问题，正值表示更积极的教师行为或教学实践
     - EDUSHORT      ：Educational Shortage (教育资源短缺)
         - 衡量学校教育资源(如教材、设施、技术等)的短缺程度
         - 正值可能表示资源短缺更严重，负值表示资源充足
     - STAFFSHORT    ： Staff Shortage (教职员工短缺)
         - 衡量学校教职员工(包括教师和行政人员)的短缺程度
         - 正值可能表示人员短缺更严重，负值表示人员配置充足

   - STUBEHA 与 TEACHBEHA 之间相关系数较高的解释：

     1. 互相影响的教育环境：

         - 教师行为对学生行为有直接影响，良好的教师行为和教学方式（高TEACHBEHA值）往往能促进积极的学生行为（高STUBEHA值）
         - 同样，学生的行为状况也会反过来影响教师的教学行为和态度
     2. 学校整体氛围因素：

         - 二者可能共同受到学校管理、校园文化和整体教育氛围的影响
         - 拥有良好管理结构的学校可能同时展现出良好的教师行为和学生行为
     3. 社区和背景因素：

         - 学校所处社区的社会经济状况可能同时影响教师和学生的行为表现
         - 资源丰富的学校环境往往同时促进积极的师生行为

   - EDUSHORT 与 STAFFSHORT 之间相关系数较高的解释：

     1. 资源分配机制：

         - 教育资源短缺和人员短缺通常是同时发生的，反映了学校资源配置的整体水平
         - 财政资源不足的学校往往既缺乏教材设备，也难以招聘和留住足够的教职员工
         
     2. 教育投入的整体性：

         - 教育投入通常是系统性的，资金充足的学校既投入于物质资源也投入于人力资源
         - 教育资源投入不足通常体现在多个方面，而非单一维度
         
     3. 累积效应：

         - 长期的资源短缺会导致教师流失，形成恶性循环
         - 同样，教职工短缺也会影响教育资源的有效利用和维护
         
     4. 区域差异因素：

         - 这种相关性可能反映了不同地区教育资源分配的不均衡性
         
         - 经济发达地区的学校往往在两个维度上都表现较好，而欠发达地区则可能两方面都面临挑战
         
           

7. **Q7 提取子表并对缺失值进行均值填补**

​	代码实现：

```py
df1=df[['PRIVATESCH','EDUSHORT','STAFFSHORT']]

# 特征PRVATESCH为先验条件,对其余各特征中可能存在的缺失值进行均值填补。
df1['EDUSHORT'] = df1.groupby('PRIVATESCH')['EDUSHORT'].transform(lambda x: x.fillna(x.mean()))
df1['STAFFSHORT'] = df1.groupby('PRIVATESCH')['STAFFSHORT'].transform(lambda x: x.fillna(x.mean()))
```



### 3.2 数据可视化

使用Matplotlib和Seaborn实现多种可视化：

1. **Q1 散点图分析STUBEHA与TEACHBEHA的关系**

   代码实现：

   ```py
   import matplotlib.pyplot as plt
   import matplotlib as mpl
   import numpy as np
   
   # 设置中文字体支持
   plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
   plt.rcParams['axes.unicode_minus'] = False
   
   # 创建散点图
   fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
   
   # 首先过滤掉缺失值和无穷值
   valid_data = df[['STUBEHA', 'TEACHBEHA']].dropna()
   valid_data = valid_data[~np.isinf(valid_data['STUBEHA']) & ~np.isinf(valid_data['TEACHBEHA'])]
   
   # 确认数据有效且足够拟合
   if len(valid_data) > 1:
       # 计算相关系数
       corr = valid_data['STUBEHA'].corr(valid_data['TEACHBEHA'])
       
       # 绘制散点图
       scatter = ax.scatter(valid_data['STUBEHA'], valid_data['TEACHBEHA'], 
                         c=np.abs(valid_data['STUBEHA'] + valid_data['TEACHBEHA']),
                         cmap='viridis',
                         alpha=0.7,
                         s=30,
                         edgecolor='w',
                         linewidth=0.2)
       
       # 安全地添加回归线
       try:
           # 使用有效数据拟合
           z = np.polyfit(valid_data['STUBEHA'], valid_data['TEACHBEHA'], 1)
           p = np.poly1d(z)
           x_sorted = sorted(valid_data['STUBEHA'])
           ax.plot(x_sorted, p(x_sorted), 'r--', linewidth=2, alpha=0.8)
           
           # 显示相关系数
           title_text = f'学生行为与教师行为相关性分析\n相关系数: {corr:.3f}'
       except np.linalg.LinAlgError:
           # 如果仍然失败，则不绘制回归线
           title_text = f'学生行为与教师行为相关性分析\n相关系数: {corr:.3f} (回归线拟合失败)'
       except Exception as e:
           title_text = f'学生行为与教师行为相关性分析\n(绘制回归线时出错: {str(e)})'
           
       # 添加颜色条和标签
       cbar = plt.colorbar(scatter, ax=ax)
       cbar.set_label('行为指标绝对值和', fontsize=10)
       
       # 设置标题和标签
       ax.set_title(title_text, fontsize=14)
       ax.set_xlabel('学生行为 (STUBEHA)', fontsize=12)
       ax.set_ylabel('教师行为 (TEACHBEHA)', fontsize=12)
       
       # 添加网格和边框美化
       ax.grid(True, linestyle='--', alpha=0.3)
       ax.spines['top'].set_visible(False)
       ax.spines['right'].set_visible(False)
   else:
       ax.text(0.5, 0.5, '有效数据不足以绘制图形', 
               horizontalalignment='center', verticalalignment='center')
   
   # 调整布局
   plt.tight_layout()
   
   # 保存图片
   plt.savefig('stubeha_teachbeha_correlation.png', dpi=300)
   plt.show()
   ```

   输出：

   <img src="/Users/halo/Desktop/数据分析及实践实验/实验3/stubeha_teachbeha_correlation.png" alt="stubeha_teachbeha_correlation" style="zoom:24%;" />

2. **Q2 饼图展示学校类型分布**

   代码实现：

   ```py
   import matplotlib.pyplot as plt
   import matplotlib.colors as mcolors
   import numpy as np
   import seaborn as sns
   
   # 设置字体和风格
   plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS中文字体支持
   plt.rcParams['axes.unicode_minus'] = False
   plt.style.use('seaborn-v0_8-pastel')  # 使用更现代的风格
   
   # 创建图形和轴，增加大小
   fig, ax = plt.subplots(figsize=(10, 8), dpi=100, facecolor='white')
   
   # 获取数据
   school_counts = df['PRIVATESCH'].value_counts()
   labels = []
   for idx in school_counts.index:
       if idx == 1:
           labels.append('私立学校')
       elif idx == 2:
           labels.append('公立学校')
       else:
           labels.append(f'类型 {idx}')
   
   # 准备颜色方案 - 使用更好看的调色板
   colors = sns.color_palette('Blues_r', len(school_counts))
   
   # 突出显示最大的部分，但不使用阴影
   explode = [0.05 if i == school_counts.argmax() else 0 for i in range(len(school_counts))]
   
   # 绘制饼图，自定义外观 - 移除shadow参数
   wedges, texts, autotexts = ax.pie(
       school_counts,
       labels=None,  # 不在图上直接显示标签
       autopct='%1.1f%%',
       startangle=90,  # 从顶部开始
       explode=explode,
       colors=colors,
       wedgeprops={'edgecolor': 'w', 'linewidth': 2, 'antialiased': True},
       textprops={'fontsize': 12, 'fontweight': 'bold', 'color': 'black'},
   )
   
   # 增强百分比文本
   for autotext in autotexts:
       autotext.set_color('black')
       autotext.set_fontsize(12)
       autotext.set_fontweight('bold')
   
   # 添加带有圆角的图例(不使用阴影)
   legend = ax.legend(
       wedges, 
       labels,
       title='学校类型',
       loc='center left',
       bbox_to_anchor=(1, 0.5),
       frameon=True,
       fancybox=True,
       fontsize=12
   )
   legend.get_title().set_fontweight('bold')
   legend.get_title().set_fontsize(14)
   
   # 设置标题，使用更大更粗的字体
   plt.title('学校类型分布', fontsize=18, fontweight='bold', pad=20)
   
   # 添加圆形边框，让饼图像甜甜圈
   central_circle = plt.Circle((0, 0), 0.45, color='white')
   fig.gca().add_artist(central_circle)  # 中心白色圆圈，创建甜甜圈效果
   
   # 添加数据标签
   total = sum(school_counts)
   ax.text(0, 0, f'总数\n{total}',
           horizontalalignment='center',
           verticalalignment='center',
           fontsize=16, fontweight='bold')
   
   # 为了增强视觉效果(不使用阴影)，可增加边缘对比度
   for wedge in wedges:
       wedge.set_edgecolor('white')
       wedge.set_linewidth(1.5)
   
   # 调整布局，确保图例不被裁剪
   plt.tight_layout()
   
   # 保存高清图片
   plt.savefig('school_type_distribution.png', dpi=300, bbox_inches='tight')
   
   plt.show()
   ```

   输出：

   <img src="/Users/halo/Desktop/数据分析及实践实验/实验3/school_type_distribution.png" alt="school_type_distribution" style="zoom:24%;" />

3. **Q3 热力图显示特征间相关系数**

   代码实现：

   ```py
   import matplotlib.pyplot as plt
   import seaborn as sns
   import numpy as np
   
   # 设置中文字体支持
   plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS中文字体
   plt.rcParams['axes.unicode_minus'] = False
   
   # 设置图形大小和分辨率
   plt.figure(figsize=(12, 10), dpi=100)
   
   # 计算相关矩阵
   corr_matrix = selected.corr(method='pearson')
   
   # 创建蒙版，隐藏上三角区域(避免重复信息)
   mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
   
   # 设置自定义颜色映射，更加醒目且色彩协调
   cmap = sns.diverging_palette(230, 20, as_cmap=True)
   
   # 绘制热力图，增强可读性
   heatmap = sns.heatmap(
       corr_matrix, 
       mask=mask,
       annot=True,           # 显示数值
       fmt=".3f",            # 保留三位小数
       cmap=cmap,            # 使用自定义色彩
       vmin=-1, vmax=1,      # 固定色标范围
       center=0,             # 确保0值为中性色
       square=True,          # 使单元格为正方形
       linewidths=.5,        # 单元格边框宽度
       cbar_kws={"shrink": .8, "label": "相关系数值"},  # 调整色标
       annot_kws={"size": 17}  # 数字标注大小
   )
   
   # 设置坐标轴标签字体大小
   plt.xticks(fontsize=10, rotation=45, ha='right')
   plt.yticks(fontsize=10)
   
   # 添加标题和边距
   plt.title("'STUBEHA', 'TEACHBEHA', 'EDUSHORT', 'STAFFSHORT' 特征之间的相关系数热力图",
              fontsize=16, pad=20, fontweight='bold')
   plt.tight_layout()
   
   # 增强图表边框
   ax = plt.gca()
   for _, spine in ax.spines.items():
       spine.set_visible(True)
       spine.set_color('black')
       spine.set_linewidth(1)
   
   # 为热力图添加整体边框
   plt.box(on=True)
   
   # 保存高清图片
   plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
   
   # 显示图形
   plt.show()
   ```

   输出：

   <img src="/Users/halo/Desktop/数据分析及实践实验/实验3/correlation_heatmap.png" alt="correlation_heatmap" style="zoom:24%;" />

### 3.3 分布检验

通过频数直方图和Q-Q图检验特征的分布特性：

1. **Q1 绘制STUBEHA和TEACHBEHA的频数直方图与理论正态曲线**

   代码实现：

   ```py
   import matplotlib.pyplot as plt
   import matplotlib as mpl
   import numpy as np
   import pandas as pd
   import seaborn as sns
   import matplotlib.font_manager as fm
   # 选择需要分析的两个特征
   df2 = df[['STUBEHA', 'TEACHBEHA']].dropna()
   
   # 创建2x1子图布局，更大的图形尺寸
   fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
   # 避免使用中文标题，改用英文标题
   fig.suptitle('Frequency Distribution Analysis of STUBEHA and TEACHBEHA', fontsize=16, fontweight='bold', y=1.05)
   
   # 颜色设置 - 使用更专业的配色
   colors = ['#3498db', '#2ecc71']
   feature_names = ['Student Behavior (STUBEHA)', 'Teacher Behavior (TEACHBEHA)']
   
   # 设置要进行比较的理论正态分布
   features = [df2['STUBEHA'], df2['TEACHBEHA']]
   
   for i, (ax, feature, name, color) in enumerate(zip(axes, features, feature_names, colors)):
       # 绘制直方图
       counts, bins, patches = ax.hist(feature, bins=10, alpha=0.7, color=color, 
                                     edgecolor='black', linewidth=1)
       
       # 添加核密度估计曲线
       sns.kdeplot(feature, ax=ax, color='#e74c3c', linewidth=2)
       
       # 计算理论正态分布曲线参数
       mu, sigma = feature.mean(), feature.std()
       x = np.linspace(feature.min(), feature.max(), 100)
       
       # 手动计算正态分布PDF
       y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
       # 缩放PDF以匹配直方图高度
       y = y * len(feature) * (bins[1] - bins[0])
       
       # 绘制理论正态分布曲线 - 使用纯英文标签
       ax.plot(x, y, 'k--', linewidth=2, label=f'Normal Dist.\nμ={mu:.2f}, σ={sigma:.2f}')
       
       # 计算偏度和峰度
       skewness = ((feature - feature.mean())**3).mean() / feature.std()**3
       kurtosis = ((feature - feature.mean())**4).mean() / feature.std()**4 - 3
       
       # 设置图表标题和轴标签 - 使用纯英文
       # 更合理的正态性评估
       normal_assessment = "Likely normal" if abs(skewness) < 1 and abs(kurtosis) < 2 else \
                           "Moderately non-normal" if abs(skewness) < 2 and abs(kurtosis) < 4 else \
                           "Highly non-normal"    
       ax.set_title(f'{name}\nNormality assessment: {normal_assessment}', fontsize=12, pad=10)
       ax.set_xlabel(name, fontsize=11)
       ax.set_ylabel('Frequency', fontsize=11)
       
       # 美化图表
       ax.spines['top'].set_visible(False)
       ax.spines['right'].set_visible(False)
       ax.grid(True, alpha=0.3)
       ax.legend(loc='upper right')
       
       # 添加垂直线显示均值和中位数 - 使用纯英文标签
       ax.axvline(mu, color='red', linestyle='-', alpha=0.7, linewidth=1.5, 
                 label=f'Mean: {mu:.2f}')
       ax.axvline(feature.median(), color='green', linestyle='--', alpha=0.7, linewidth=1.5,
                 label=f'Median: {feature.median():.2f}')
       
       # 标注一些关键统计数据 - 使用纯英文
       props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
       textstr = f'Mean: {mu:.3f}\nStd Dev: {sigma:.3f}\nKurtosis: {kurtosis:.3f}\nSkewness: {skewness:.3f}'
       ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
              verticalalignment='top', bbox=props)
   
   # 调整子图之间的间距
   plt.tight_layout()
   plt.savefig('normality_analysis_english.png', dpi=300, bbox_inches='tight')
   plt.show()
   ```

   输出：

   <img src="/Users/halo/Desktop/数据分析及实践实验/实验3/normality_analysis_english.png" alt="normality_analysis_english" style="zoom:24%;" />

   如图所示，经过绘制两特征的频数直方图，直观上两特征均近似服从正态分布

   进一步通过计算两特征的 偏度 和 峰度 ，得到二者均近似服从正态分布的结论

   

2. **Q2 使用Q-Q图进一步验证正态性**

   代码实现：

   ```py
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import statsmodels.api as sm
   
   # 设置样式和字体
   plt.style.use('seaborn-v0_8-whitegrid')
   plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
   plt.rcParams['axes.unicode_minus'] = False
   
   # 准备数据：移除缺失值
   df2 = df[['STUBEHA', 'TEACHBEHA']].dropna()
   
   # 创建图形
   fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
   fig.suptitle('Q-Q Plot: Normality Assessment', fontsize=16, fontweight='bold')
   
   # 颜色设置
   colors = ['#3498db', '#2ecc71']
   feature_names = ['Student Behavior (STUBEHA)', 'Teacher Behavior (TEACHBEHA)']
   features = [df2['STUBEHA'], df2['TEACHBEHA']]
   
   # 绘制每个特征的Q-Q图
   for i, (ax, feature, name, color) in enumerate(zip(axes, features, feature_names, colors)):
       # 使用statsmodels绘制QQ图 - 大大简化了代码
       sm.qqplot(feature, line='45', ax=ax, markerfacecolor=color, alpha=0.7)
       
       # 计算正态性指标
       skewness = ((feature - feature.mean())**3).mean() / feature.std()**3
       kurtosis = ((feature - feature.mean())**4).mean() / feature.std()**4 - 3
       
       # 评估正态性
       if abs(skewness) < 1 and abs(kurtosis) < 2:
           normal_assessment = "Likely normal"
       elif abs(skewness) < 2 and abs(kurtosis) < 4:
           normal_assessment = "Moderately non-normal"
       else:
           normal_assessment = "Highly non-normal"
       
       # 设置标题和标签
       ax.set_title(f'Q-Q Plot: {name}\nAssessment: {normal_assessment}', fontsize=12)
       ax.set_xlabel('Theoretical Quantiles', fontsize=10)
       ax.set_ylabel('Sample Quantiles', fontsize=10)
       
       # 添加统计信息文本框
       stats_text = f'Skewness: {skewness:.3f}\nKurtosis: {kurtosis:.3f}'
       ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
       
       # 美化坐标轴
       ax.spines['top'].set_visible(False)
       ax.spines['right'].set_visible(False)
       ax.grid(True, alpha=0.3)
       
       # 设置轴刻度字体大小
       ax.tick_params(axis='both', labelsize=9)
   
   # 调整布局
   plt.tight_layout()
   plt.savefig('qq_plot_statsmodels.png', dpi=300, bbox_inches='tight')
   plt.show()
   ```

   输出：

   <img src="/Users/halo/Desktop/数据分析及实践实验/实验3/qq_plot_statsmodels.png" alt="qq_plot_statsmodels" style="zoom:24%;" />

   由输出图像可以看出，两特征近似服从正态分布

   

   两种方法的结合使用让我有以下感想：

   1. **多角度分析的重要性**：单一可视化方法往往无法提供完整的数据特征。

      直方图：通过视觉上的分布形态（如钟形曲线）粗略判断是否接近正态分布

      Q-Q图：通过数据点是否沿对角线分布来更严格地检验是否符合正态分布

   2. **统计假设的谨慎验证**：肉眼判断分布是否正态可能存在偏差，Q-Q图提供了更严格的检验方法。

   

3. **Q3 绘制特征间Q-Q图比较两特征分布一致性**

   代码实现：

   ```py
   import matplotlib.pyplot as plt
   import numpy as np
   import matplotlib.font_manager as fm
   
   # 设置样式
   plt.style.use('seaborn-v0_8-whitegrid')
   
   # 过滤缺失值
   df2 = df[['STUBEHA', 'TEACHBEHA']].dropna()
   
   # 创建图形
   plt.figure(figsize=(8, 7))
   
   # 步骤1: 对两个特征的值分别排序
   sorted_stubeha = np.sort(df2['STUBEHA'])
   sorted_teachbeha = np.sort(df2['TEACHBEHA'])
   
   # 步骤2: 绘制特征-特征 Q-Q 图
   plt.scatter(sorted_stubeha, sorted_teachbeha, alpha=0.7, color='#3498db')
   
   # 步骤3: 添加 y=x 参考线
   min_val = min(sorted_stubeha.min(), sorted_teachbeha.min())
   max_val = max(sorted_stubeha.max(), sorted_teachbeha.max())
   plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
   
   # 添加描述统计量
   stubeha_mean = df2['STUBEHA'].mean()
   teachbeha_mean = df2['TEACHBEHA'].mean()
   stubeha_std = df2['STUBEHA'].std()
   teachbeha_std = df2['TEACHBEHA'].std()
   
   # 标注信息 - 使用英文
   stats_text = (f"STUBEHA: Mean={stubeha_mean:.2f}, SD={stubeha_std:.2f}\n"
                 f"TEACHBEHA: Mean={teachbeha_mean:.2f}, SD={teachbeha_std:.2f}")
   
   plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
               ha='left', va='top', fontsize=10)
   
   # 设置标题和标签 - 使用英文
   plt.title('Distribution Consistency Q-Q Plot: STUBEHA vs TEACHBEHA', fontsize=14)
   plt.xlabel('STUBEHA Quantiles', fontsize=12)
   plt.ylabel('TEACHBEHA Quantiles', fontsize=12)
   plt.grid(True, alpha=0.3)
   plt.legend()
   
   # 美化图形
   plt.grid(True, linestyle='--', alpha=0.3)
   for spine in plt.gca().spines.values():
       spine.set_visible(False)
   
   plt.tight_layout()
   plt.savefig('feature_feature_qqplot.png', dpi=300, bbox_inches='tight')
   plt.show()
   ```

   输出：

   <img src="/Users/halo/Desktop/数据分析及实践实验/实验3/feature_feature_qqplot.png" alt="feature_feature_qqplot" style="zoom:24%;" />

​	结论：

​		这说明 STUBEHA 和 TEACHBEHA 特征在 中间和右侧 的分布相似性较高， 在左侧分布有较大差异

​	

### 3.4 参数估计

基于正态分布假设，对STUBEHA和TEACHBEHA进行参数估计：

1. **Q1 使用极大似然法估计均值和方差参数 & 比较最大似然估计与无偏估计的差异**

   代码实现：

   ```py
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   from scipy import stats
   
   # 准备数据：过滤缺失值
   df_clean = df[['STUBEHA', 'TEACHBEHA']].dropna()
   
   # 计算参数估计
   results = pd.DataFrame(index=['STUBEHA', 'TEACHBEHA'])
   
   # 计算样本大小
   n_stubeha = len(df_clean['STUBEHA'])
   n_teachbeha = len(df_clean['TEACHBEHA'])
   
   # 最大似然估计 (MLE)
   mu_mle_stubeha = df_clean['STUBEHA'].mean()
   mu_mle_teachbeha = df_clean['TEACHBEHA'].mean()
   
   # 方差的最大似然估计 (分母为n)
   var_mle_stubeha = df_clean['STUBEHA'].var(ddof=0)  # ddof=0 表示分母为n
   var_mle_teachbeha = df_clean['TEACHBEHA'].var(ddof=0)
   
   # 无偏方差估计 (分母为n-1)
   var_unbiased_stubeha = df_clean['STUBEHA'].var(ddof=1)  # ddof=1 表示分母为n-1
   var_unbiased_teachbeha = df_clean['TEACHBEHA'].var(ddof=1)
   
   # 创建结果表格
   results['样本大小'] = [n_stubeha, n_teachbeha]
   results['均值 (MLE估计)'] = [mu_mle_stubeha, mu_mle_teachbeha]
   results['方差 (MLE估计, 分母=n)'] = [var_mle_stubeha, var_mle_teachbeha]
   results['方差 (无偏估计, 分母=n-1)'] = [var_unbiased_stubeha, var_unbiased_teachbeha]
   results['标准差 (MLE估计)'] = [np.sqrt(var_mle_stubeha), np.sqrt(var_mle_teachbeha)]
   results['偏度'] = [stats.skew(df_clean['STUBEHA']), stats.skew(df_clean['TEACHBEHA'])]
   results['峰度'] = [stats.kurtosis(df_clean['STUBEHA']), stats.kurtosis(df_clean['TEACHBEHA'])]
   
   # 显示结果
   print("基于正态分布假设的参数估计：")
   print(results.round(4))
   
   # 计算MLE与无偏估计的差异百分比
   bias_percent_stubeha = (var_unbiased_stubeha - var_mle_stubeha) / var_unbiased_stubeha * 100
   bias_percent_teachbeha = (var_unbiased_teachbeha - var_mle_teachbeha) / var_unbiased_teachbeha * 100
   
   print("\n方差估计的偏差：")
   print(f"STUBEHA方差的MLE估计比无偏估计小 {bias_percent_stubeha:.4f}%")
   print(f"TEACHBEHA方差的MLE估计比无偏估计小 {bias_percent_teachbeha:.4f}%")
   print(f"理论上的偏差应为 {100/n_stubeha:.4f}% (1/n)")
   
   # 可视化：正态分布拟合
   plt.figure(figsize=(15, 6))
   
   # STUBEHA
   plt.subplot(1, 2, 1)
   sns.histplot(df_clean['STUBEHA'], kde=True, stat='density', color='skyblue', alpha=0.6)
   
   # 添加正态分布拟合曲线
   x = np.linspace(df_clean['STUBEHA'].min(), df_clean['STUBEHA'].max(), 100)
   y = stats.norm.pdf(x, mu_mle_stubeha, np.sqrt(var_mle_stubeha))
   plt.plot(x, y, 'r-', linewidth=2, label=f'Normal: μ={mu_mle_stubeha:.2f}, σ²={var_mle_stubeha:.2f}')
   
   plt.title('STUBEHA: Normal Distribution Fit (MLE)')
   plt.xlabel('Value')
   plt.ylabel('Density')
   plt.legend()
   
   # TEACHBEHA
   plt.subplot(1, 2, 2)
   sns.histplot(df_clean['TEACHBEHA'], kde=True, stat='density', color='lightgreen', alpha=0.6)
   
   # 添加正态分布拟合曲线
   x = np.linspace(df_clean['TEACHBEHA'].min(), df_clean['TEACHBEHA'].max(), 100)
   y = stats.norm.pdf(x, mu_mle_teachbeha, np.sqrt(var_mle_teachbeha))
   plt.plot(x, y, 'r-', linewidth=2, label=f'Normal: μ={mu_mle_teachbeha:.2f}, σ²={var_mle_teachbeha:.2f}')
   
   plt.title('TEACHBEHA: Normal Distribution Fit (MLE)')
   plt.xlabel('Value')
   plt.ylabel('Density')
   plt.legend()
   
   plt.tight_layout()
   plt.savefig('normal_distribution_parameter_estimation.png', dpi=300, bbox_inches='tight')
   plt.show()
   ```

   输出：

   <img src="/Users/halo/Library/Application Support/typora-user-images/image-20250403195844431.png" alt="image-20250403195844431" style="zoom:50%;" />

   <img src="/Users/halo/Desktop/数据分析及实践实验/实验3/normal_distribution_parameter_estimation.png" alt="normal_distribution_parameter_estimation" style="zoom:24%;" />

   所以在上述极大似然估计中，均值是无偏估计，方差不是无偏估计

   

2. **Q2 对 STUBEHA 进行常数最小二乘估计**

   代码实现：

   ```py
   import pandas as pd
   import numpy as np
   from scipy.optimize import minimize
   
   # 提取STUBEHA数据
   stubeha_data = df['STUBEHA'].dropna().values
   
   # 定义最小二乘损失函数
   def mse_loss(a, data):
       return np.sum((data - a) ** 2)
   
   # 使用优化方法求解最小二乘估计值
   initial_guess = 0.0  # 初始猜测值
   result = minimize(mse_loss, initial_guess, args=(stubeha_data,))  # 注意这里添加了逗号
   ls_estimate = result.x[0]
   
   print(f"最小二乘估计值（优化求解）: {ls_estimate:.6f}")
   
   # 验证与样本均值是否相等
   print(f"样本均值: {np.mean(stubeha_data):.6f}")
   print(f"差异: {abs(ls_estimate - np.mean(stubeha_data)):.10f}")
   ```
   
   输出：
   
   <img src="/Users/halo/Library/Application Support/typora-user-images/image-20250403201251711.png" alt="image-20250403201251711" style="zoom:50%;" />
   
   

### 3.5 假设检验

对STUBEHA和TEACHBEHA的均值差异进行检验：

1. **Q1 检验方法的选择**：

   - ##### 检验方法选择：成对检验
     理由如下：

     1. 数据结构特点：每一行数据代表同一所学校的观测值，STUBEHA和TEACHBEHA是针对同一学校测量的两个不同特征
     2. 样本依赖性：两个特征值之间存在自然配对关系，它们来自同一观测单位（学校）
     3. 统计效率：成对检验通过控制"学校"这一潜在混杂因素，能够更精确地检测均值差异

   - **单侧检验原假设:**

     假设我们想检验学生行为(STUBEHA)的均值是否小于教师行为(TEACHBEHA)的均值，则原假设为：

     H₀: μ₁ - μ₂ ≥ 0 (STUBEHA的均值不小于TEACHBEHA的均值) 

     H₁: μ₁ - μ₂ < 0 (STUBEHA的均值小于TEACHBEHA的均值)

     注：如果想检验STUBEHA均值是否大于TEACHBEHA均值，原假设则为μ₁ - μ₂ ≤ 0。

     

2. 设定假设：H₀: μ₁ - μ₂ ≥ 0，H₁: μ₁ - μ₂ < 0

   ```py
   import scipy.stats as stats
   import numpy as np
   import matplotlib.pyplot as plt
   
   # 提取非缺失数据对
   data = df[['STUBEHA', 'TEACHBEHA']].dropna()
   
   # 计算差值
   diff = data['STUBEHA'] - data['TEACHBEHA']
   
   # 描述性统计
   print("描述性统计：")
   print(f"STUBEHA均值: {data['STUBEHA'].mean():.4f}, 标准差: {data['STUBEHA'].std():.4f}")
   print(f"TEACHBEHA均值: {data['TEACHBEHA'].mean():.4f}, 标准差: {data['TEACHBEHA'].std():.4f}")
   print(f"差值均值: {diff.mean():.4f}, 差值标准差: {diff.std():.4f}")
   print(f"样本大小: {len(data)}")
   
   # 执行成对t检验
   t_stat, p_value_two_sided = stats.ttest_rel(data['STUBEHA'], data['TEACHBEHA'])
   
   # 计算单侧p值 (H1: STUBEHA < TEACHBEHA)
   p_value_one_sided = p_value_two_sided / 2 if t_stat < 0 else 1 - p_value_two_sided / 2
   
   # 打印结果并改进了非常小的p值的格式
   print("\n假设检验结果：")
   print(f"t统计量: {t_stat:.4f}")
   
   # 使用科学计数法显示非常小的p值
   if p_value_two_sided < 0.0001:
       print(f"双侧p值: {p_value_two_sided:.4e}")
   else:
       print(f"双侧p值: {p_value_two_sided:.6f}")
   
   if p_value_one_sided < 0.0001:
       print(f"单侧p值 (H1: STUBEHA < TEACHBEHA): {p_value_one_sided:.4e}")
   else:
       print(f"单侧p值 (H1: STUBEHA < TEACHBEHA): {p_value_one_sided:.6f}")
   
   # 结论
   alpha = 0.05
   print("\n检验结论：")
   if p_value_one_sided < alpha:
       print(f"在显著性水平α={alpha}下，拒绝原假设")
       print("有足够证据表明学生行为(STUBEHA)的均值显著小于教师行为(TEACHBEHA)的均值")
   else:
       print(f"在显著性水平α={alpha}下，不能拒绝原假设")
       print("没有足够证据表明学生行为(STUBEHA)的均值显著小于教师行为(TEACHBEHA)的均值")
   
   # 可视化
   plt.figure(figsize=(10, 6))
   plt.hist(diff, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
   plt.axvline(diff.mean(), color='red', linestyle='--', linewidth=2, 
              label=f'均值差异: {diff.mean():.4f}')
   plt.axvline(0, color='green', linestyle='-', linewidth=2, 
              label='零差异线 (H₀)')
   
   plt.title('STUBEHA - TEACHBEHA 差异分布', fontsize=14)
   plt.xlabel('差异', fontsize=12)
   plt.ylabel('频数', fontsize=12)
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.savefig('paired_ttest_distribution.png', dpi=300)
   plt.show()
   ```

   输出：

   <img src="/Users/halo/Library/Application Support/typora-user-images/image-20250421161648484.png" alt="image-20250421161648484" style="zoom:50%;" />

   <img src="/Users/halo/Library/Application Support/typora-user-images/image-20250403202832954.png" alt="image-20250403202832954" style="zoom:50%;" />

   

3. **Q3 叙述由 Q2 所得到的结论**

   根据成对t检验的结果，我们可以得出以下结论：

   - 统计显著性判断：

     假设检验结果显示t统计量为负值，且对应的单侧p值小于0.05的显著性水平

     因此，我们有足够的统计证据拒绝原假设(H₀: μ₁ - μ₂ ≥ 0)

     接受备择假设，即学生行为( STUBEHA )的均值 显著小于 教师行为( TEACHBEHA )的均值

   - 效应大小考量：

     两个特征的均值差异不仅具有统计显著性，还需考虑实际效应大小

     差值的均值（STUBEHA - TEACHBEHA）为负值，表明平均而言，学校中学生行为的评分低于教师行为评分

     效应大小（Cohen's d = 差值均值/差值标准差）处于中等水平，表明这种差异不仅具有统计意义，也具有实际意义

   - 实际教育意义：

     这种差异表明在同一学校环境中，学生行为问题通常比教师行为问题更为显著

     学校中可能存在的师生行为不一致性，可能反映了校园环境中的行为标准差异或评估偏差

     这种差异可能源于学校管理方式、教育政策或社会文化因素的影响

   - 方法论考量：

     成对t检验是适合的分析方法，因为它控制了"学校"这一重要变量。通过比较同一学校内的师生行为差异，排除了不同学校间可能存在的混杂因素

     这种方法提高了检验的统计效力，使我们能够更精确地检测均值差异

     

   4. **Q4 上述结论隐含了犯哪一类错误的可能?相应犯错概率是多少?**

      

      答：隐含了犯第一类错误的概率，概率为 5%

   

## 4. 实验结果与分析

### 4.1 数据预处理结果

- 数据集包含21903行和197列
- 识别到"CYC"和"ADMINMODE"两个冗余列，值分别为"07MS"和"2"
- PRIVATESCH列包含四种取值："PUBLIC"(12234)、"MISSING"(5295)、"PRIVATE"(3527)和"INVALID"(251)

### 4.2 描述性统计结果

| 特征       | 均值   | 标准差 | 最小值 | 中位数 | 最大值 |
| ---------- | ------ | ------ | ------ | ------ | ------ |
| STUBEHA    | 0.042  | 1.237  | -4.354 | 0.042  | 3.627  |
| TEACHBEHA  | 0.108  | 1.158  | -3.239 | 0.227  | 3.834  |
| EDUSHORT   | 0.121  | 1.091  | -1.932 | 0.100  | 3.523  |
| STAFFSHORT | -0.014 | 1.060  | -2.589 | 0.013  | 4.113  |

### 4.3 相关性分析

特征间的相关关系：

- STUBEHA与TEACHBEHA: r = 0.634
- EDUSHORT与STAFFSHORT: r = 0.484
- STUBEHA与EDUSHORT: r = 0.240
- STUBEHA与STAFFSHORT: r = 0.257
- TEACHBEHA与EDUSHORT: r = 0.215
- TEACHBEHA与STAFFSHORT: r = 0.332

这表明：

1. 学生行为与教师行为高度相关，可能互相影响或受共同因素影响
2. 教育资源短缺与教职员工短缺中度相关，反映了学校资源配置的整体性

### 4.4 分布检验结果

- STUBEHA特征：根据偏度和峰度分析，可能符合正态分布(Likely normal)
- TEACHBEHA特征：分布呈现双峰特性，不符合正态分布(Moderately non-normal)
- 两特征对比：通过Q-Q图显示，两特征在中间和右侧分布相似性较高，左侧差异较大

### 4.5 参数估计结果

极大似然估计结果：

- STUBEHA: μ = 0.042, σ² = 1.529
- TEACHBEHA: μ = 0.108, σ² = 1.341

均值的最大似然估计是无偏的，但方差的最大似然估计有偏，比无偏估计小约0.0048%。

### 4.6 假设检验结果

成对t检验结果：

- t统计量:  -9.4206
- 单侧p值:2.4682e-21
- 在α=0.05显著性水平下，拒绝原假设
- 结论：有足够证据表明学生行为(STUBEHA)的均值显著小于教师行为(TEACHBEHA)的均值
- 犯第一类错误的概率为5%

## 5. 结论

1. 数据分析表明，学生行为与教师行为之间存在显著正相关，教育资源短缺与教职员工短缺也呈现中度正相关。这表明学校环境因素之间存在明显的相互关联性。
2. 分布检验显示，STUBEHA特征可能近似服从正态分布，而TEACHBEHA特征则表现出非正态特性。
3. 参数估计结果证实，均值的极大似然估计是无偏的，而方差的极大似然估计则存在微小偏差。
4. 假设检验结果表明，在同一学校环境中，学生行为问题通常比教师行为问题更为显著，这可能反映了校园环境中的行为标准差异或评估偏差。
5. 公立学校在数据集中占绝大多数，这也可能影响整体统计结果的代表性和可推广性。

这些发现对理解学校教育环境因素之间的关系及其对教育质量的影响具有重要意义，可为教育政策制定和学校管理提供数据支持。

