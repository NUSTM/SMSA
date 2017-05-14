#### corpus(文件太大无法上传)：
- 中文自然标注微博预料
- 英文自然标注微博语料

#### data：

- 中英文多个标准数据集（带标签）

#### ML-TOOLS：

- 机器学习工具包

#### 该份代码可以：

- 可以用于构建情感词典
- 完成一般的情感分类问题，包含了预处理、模型构建、特征工程、交叉验证、性能分析等多个模块
- 可以进行bagging集成学习




#### 分别根据MI、IG、WLLR、CHI四种方式构建情感词典，具体参见代码主函数部分

- gen_universal_lexicon_mi.py
- gen_universal_lexicon_ig.py
- gen_universal_lexicon_wllr.py
- gen_universal_lexicon_chi.py



#### 情感分析主类:  Mbsa_new.py:具体使用方式参见代码主函数

- 可以设置基础特征：词、词性的n-gram (n=1,2,3...)
- 进行N折交叉验证：参考cv_main.py



#### 基于词典的规则情感分析主类：Lbsa.py



#### 性能评估类：performance.py

包含多种评估标准计算函数



#### 文本处理工具：pytc.py

其中的build_samps() 函数可以用来自定义添加额外的多种特征


#### Bagging集成学习：random_main.py

投票、概率平均

#### 随机选择样本： pick_samps.py
