**Github/invest_model README**

---

# Invest Model Project

欢迎来到Invest Model项目！本项目旨在通过一系列精心设计的阶段模型来分析和优化投资策略。以下是快速入门指南，帮助您克隆项目环境并运行我们的模型。

## 环境搭建

### 克隆Conda环境

1. **安装Miniconda/Anaconda**  
   首先，确保您的系统上安装了[Miniconda](https://docs.conda.io/en/latest/miniconda.html)或[Anaconda](https://www.anaconda.com/products/distribution)。它们提供了conda环境管理工具。

2. **克隆环境**  
   打开终端或Anaconda Prompt，导航至项目根目录（即包含此`README.md`文件的目录）。

   ```bash
   cd /path/to/invest_model
   ```

   使用以下命令根据`environment.yml`文件创建一个新的Conda环境。我们假设您想将新环境命名为`invest-env`：

   ```bash
   conda env create -n invest-env -f environment.yml
   ```

   这条命令会自动下载并安装所有必要的Python包和依赖项，以匹配原始项目的环境设置。

3. **激活环境**  
   创建完毕后，激活新环境：

   - **Linux/macOS**:

     ```bash
     conda activate invest-env
     ```

   - **Windows**:

     ```bash
     conda activate invest-env
     ```

## 模型执行流程

本项目包含三个关键阶段的模型脚本，分别是`phase0.py`、`phase1.py`和`phase2.py`，每个阶段按顺序构建并分析投资策略的不同方面。

### 执行阶段模型

确保您当前激活的是`invest-env`环境，然后按照以下顺序依次运行各阶段脚本：

#### Phase 0

```bash
python phase0.py
```

该阶段负责数据获取。

#### Phase 1

```bash
python phase1.py
```

这一阶段进行数据处理，特征图生成。

#### Phase 2

```bash
python phase2.py
```

最后阶段为模型训练与评估。

### 工具与辅助模块

- `models/`: 包含项目中的自定义模型实现。
- `utils/`: 存放通用工具函数和辅助类，支持各阶段脚本。
- `modify.py`: 用于数据或参数调整的脚本。


---

现在，您已准备好深入探索和扩展我们的投资模型。祝您分析愉快，投资有道！