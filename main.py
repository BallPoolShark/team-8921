#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data processing and model training pipeline executor.

This module provides functionality to sequentially execute Python scripts
for data preprocessing and model training. It runs scripts from the Preprocess
folder (cell1.py to cell4.py) followed by scripts from the Model folder
(cell5.py to cell6.py) in a specified order.

The execution process includes error handling, progress reporting, and
validation to ensure all scripts exist before execution.

Example:
    To run the entire pipeline, execute this script directly::

        $ python main.py

Attributes:
    This module does not define any module-level attributes.
"""

import sys
import os
from pathlib import Path


def run_script(script_path, shared_namespace):
    """Execute a Python script file in the current interpreter context.

    This function reads and executes the specified Python script using exec().
    It provides detailed progress output and error reporting. If the script
    fails or does not exist, the program will terminate with exit code 1.

    Args:
        script_path (str or Path): The file path to the Python script to execute.
            Can be an absolute or relative path.
        shared_namespace (dict): Shared namespace dictionary for variable
            persistence across script executions.

    Raises:
        SystemExit: Exits with code 1 if the script file does not exist or
            if any exception occurs during script execution.

    Note:
        The script is executed in a shared global namespace, allowing
        variables to be passed between scripts. __name__ is set to
        '__main__' and __file__ is set to the script's path.

    Example:
        >>> namespace = {'__name__': '__main__'}
        >>> run_script('Preprocess/cell1.py', namespace)
        ============================================================
        正在执行: Preprocess/cell1.py
        ============================================================
        ...
        完成: Preprocess/cell1.py
    """
    script_path = Path(script_path)

    if not script_path.exists():
        print(f"错误: 文件不存在 - {script_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"正在执行: {script_path}")
    print(f"{'='*60}")

    try:
        # 读取并执行脚本
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()

        # 更新当前脚本的 __file__ 路径
        shared_namespace['__file__'] = str(script_path)

        # 在共享的全局命名空间中执行脚本
        exec(script_content, shared_namespace)

        print(f"\n完成: {script_path}")

    except Exception as e:
        print(f"\n错误: 执行 {script_path} 时发生异常")
        print(f"异常信息: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Execute the complete data processing and model training pipeline.

    This function serves as the entry point for the pipeline execution.
    It sequentially runs six Python scripts in the following order:
    1. Preprocess/cell1.py - Initial data preprocessing
    2. Preprocess/cell2.py - Secondary data preprocessing
    3. Preprocess/cell3.py - Tertiary data preprocessing
    4. Preprocess/cell4.py - Final data preprocessing
    5. Model/cell5.py - Model training (part 1)
    6. Model/cell6.py - Model training (part 2)

    All scripts share a common namespace, allowing variables to be passed
    between them (e.g., txn, alert, pred, feat, etc.).

    The function will terminate the entire program if any script fails
    during execution.

    Raises:
        SystemExit: Propagated from run_script() if any script fails or
            does not exist.

    Example:
        >>> main()
        开始执行数据处理和模型训练流程...
        ============================================================
        正在执行: Preprocess/cell1.py
        ============================================================
        ...
        所有脚本执行完成！
    """
    print("开始执行数据处理和模型训练流程...")

    # 获取项目根目录
    project_root = Path(__file__).parent

    # 创建共享命名空间，所有脚本将在此空间中执行
    shared_namespace = {'__name__': '__main__'}

    # 定义要执行的脚本列表
    scripts = [
        # Preprocess文件夹
        project_root / "Preprocess" / "cell1.py",
        project_root / "Preprocess" / "cell2.py",
        project_root / "Preprocess" / "cell3.py",
        project_root / "Preprocess" / "cell4.py",
        # Model文件夹
        project_root / "Model" / "cell5.py",
        project_root / "Model" / "cell6.py",
    ]

    # 依序执行每个脚本，共享命名空间
    for script in scripts:
        run_script(script, shared_namespace)

    print(f"\n{'='*60}")
    print("所有脚本执行完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
