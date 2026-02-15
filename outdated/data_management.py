import os
from nis import match

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def find_folder_parent(search_name: str, root_path: str) -> Optional[str]:
    """
    在指定路径下搜索文件夹，并返回其父文件夹名称

    Args:
        search_name: 要搜索的文件夹名称（可以是部分匹配）
        root_path: 搜索的根目录路径

    Returns:
        找到的文件夹的父文件夹名称，如果未找到则返回None
    """
    root = Path(root_path)

    if not root.exists():
        print(f"警告: 根路径不存在 - {root_path}")
        return None

    # 遍历所有子目录
    for dirpath, dirnames, filenames in os.walk(root):
        for dirname in dirnames:
            # 检查文件夹名是否匹配（支持部分匹配）
            if search_name in dirname:
                folder_path = Path(dirpath) / dirname
                parent_name = folder_path.parent.name
                print(f"找到: {dirname} -> 父文件夹: {parent_name}")
                return parent_name

    print(f"未找到匹配的文件夹: {search_name}")
    return None


def process_csv(csv_path: str,
                column_name: str,
                search_root: str,
                output_column: str = None,
                exact_match: bool = False) -> None:
    """
    处理CSV文件，查找文件夹并更新父文件夹信息

    Args:
        csv_path: CSV文件路径
        column_name: 要读取的列名
        search_root: 搜索文件夹的根目录
        output_column: 输出结果的列名（如果为None，则在原列后新建列）
        exact_match: 是否精确匹配文件夹名
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        # 如果UTF-8失败，尝试其他编码
        df = pd.read_csv(csv_path, encoding='gbk')

    # 检查列是否存在
    if column_name not in df.columns:
        raise ValueError(f"列 '{column_name}' 不存在于CSV文件中。可用列: {list(df.columns)}")

    # 确定输出列名
    if output_column is None:
        output_column = f"{column_name}_父文件夹"

    # 创建新列存储结果
    df[output_column] = None

    # 处理每一行
    print(f"\n开始处理 {len(df)} 行数据...")
    print("=" * 60)

    for idx, row in df.iterrows():
        search_value = row[column_name]

        # 跳过空值
        if pd.isna(search_value):
            print(f"第 {idx + 1} 行: 跳过（空值）")
            continue

        search_value = str(search_value).strip()

        if not search_value:
            print(f"第 {idx + 1} 行: 跳过（空字符串）")
            continue

        print(f"\n第 {idx + 1} 行: 搜索 '{search_value}'")

        # 查找文件夹
        if exact_match:
            parent_name = find_folder_parent_exact(search_value, search_root)
        else:
            parent_name = find_folder_parent(search_value, search_root)

        # 更新DataFrame
        if parent_name:
            df.at[idx, output_column] = parent_name

    # 保存结果
    output_path = csv_path.replace('.csv', '_updated.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print(f"处理完成！结果已保存至: {output_path}")
    print(f"成功找到 {df[output_column].notna().sum()} 个文件夹")
    print(f"未找到 {df[output_column].isna().sum()} 个文件夹")
def process_csv2(csv_path: str,
                column_name: str,
                reference_csv: str,  # 改名更清晰
                output_column: str = None,
                exact_match: bool = False) -> None:
    """
    处理CSV文件，从参考CSV中查找并更新信息

    Args:
        csv_path: 要处理的CSV文件路径
        column_name: 要读取的列名（用于匹配BraTS21 ID）
        reference_csv: 参考CSV文件路径（包含MGMT、IDH等信息）
        output_column: 输出结果的列名
        exact_match: 是否精确匹配
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')

    # 检查列是否存在
    if column_name not in df.columns:
        raise ValueError(f"列 '{column_name}' 不存在于CSV文件中。可用列: {list(df.columns)}")

    # 创建新列（如果不存在）
    for col in ['MGMT', '1p19q_codeleted', 'IDH_mutated', 'Histological grade', 'TERT', 'WHO']:
        if col not in df.columns:
            df[col] = None

    # 处理每一行
    print(f"\n开始处理 {len(df)} 行数据...")
    print("=" * 60)

    for idx, row in df.iterrows():
        search_value = row[column_name]

        # 跳过空值
        if pd.isna(search_value):
            print(f"第 {idx + 1} 行: 跳过（空值）")
            continue

        search_value = str(search_value).strip()

        if not search_value:
            print(f"第 {idx + 1} 行: 跳过（空字符串）")
            continue

        print(f"\n第 {idx + 1} 行: 搜索 '{search_value}'")

        # 传入df和idx，而不是row
        find_target_column(df, idx, search_value, reference_csv)

    # 保存结果
    output_path = csv_path.replace('.csv', '_updated_2.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print(f"处理完成！结果已保存至: {output_path}")

def find_folder_parent_exact(search_name: str, root_path: str) -> Optional[str]:
    """
    精确匹配文件夹名称

    Args:
        search_name: 要搜索的文件夹名称（精确匹配）
        root_path: 搜索的根目录路径

    Returns:
        找到的文件夹的父文件夹名称，如果未找到则返回None
    """
    root = Path(root_path)

    if not root.exists():
        print(f"警告: 根路径不存在 - {root_path}")
        return None

    # 遍历所有子目录
    for dirpath, dirnames, filenames in os.walk(root):
        for dirname in dirnames:
            # 精确匹配
            if dirname == search_name:
                folder_path = Path(dirpath) / dirname
                parent_name = folder_path.parent.name
                print(f"找到: {dirname} -> 父文件夹: {parent_name}")
                return parent_name

    print(f"未找到匹配的文件夹: {search_name}")
    return None


def find_target_column(df, idx, search_value: str, csv_path: str) -> None:
    """
    在CSV文件中查找匹配的行，更新DataFrame对应行的值

    Args:
        df: 要更新的DataFrame
        idx: 当前行的索引
        search_value: 要搜索的值（在'BraTS21 ID'列中查找）
        csv_path: 参考CSV文件路径
    """
    try:
        ucsf = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        ucsf = pd.read_csv(csv_path, encoding='gbk')

    # 检查搜索列是否存在
    if 'BraTS21 ID' not in ucsf.columns:
        raise ValueError(f"列 'BraTS21 ID' 不存在于CSV文件中。可用列: {list(ucsf.columns)}")

    # 查找匹配的行
    matched_rows = ucsf[ucsf['BraTS21 ID'] == search_value]

    if matched_rows.empty:
        print(f"警告: 未找到 'BraTS21 ID' = '{search_value}' 的记录")
        return

    # 获取第一个匹配行的值
    mgmt_status = matched_rows['MGMT status'].iloc[0]
    p19q_status = matched_rows['1p/19q'].iloc[0]
    idh_status = matched_rows['IDH'].iloc[0]
    who_grade = matched_rows['WHO CNS Grade'].iloc[0]

    # 直接修改DataFrame（使用.loc而不是row）
    if mgmt_status == "negative":
        df.loc[idx, 'MGMT'] = 0
    elif mgmt_status == "unknown":
        df.loc[idx, 'MGMT'] = -1
    elif mgmt_status == "positive":
        df.loc[idx, 'MGMT'] = 1

    if p19q_status == "intact":
        df.loc[idx, '1p19q_codeleted'] = 0
    elif p19q_status == "unknown":
        df.loc[idx, '1p19q_codeleted'] = -1
    else:
        df.loc[idx, '1p19q_codeleted'] = 1

    if idh_status == "wildtype":
        df.loc[idx, 'IDH_mutated'] = 0
    elif idh_status == "mutated (NOS)":
        df.loc[idx, 'IDH_mutated'] = 1
    else:
        df.loc[idx, 'IDH_mutated'] = 2

    df.loc[idx, 'Histological grade'] = -1
    df.loc[idx, 'TERT'] = -1
    df.loc[idx, 'WHO'] = who_grade


def main():
    """主函数 - 使用示例"""

    # ===== 配置参数 =====
    CSV_FILE = "BraTSfile_updated.csv"  # CSV文件路径
    COLUMN_NAME = "BraTS2021"  # 要读取的列名
    SEARCH_ROOT = "/Users/mbpm2max/Downloads/UCSF-PDGM-metadata_v5.csv"  # 搜索的根目录
    OUTPUT_COLUMN = "BraTS21 ID"  # 输出列名（可选）
    EXACT_MATCH = False  # 是否精确匹配

    # ===== 执行处理 =====
    try:
        process_csv2(
            csv_path=CSV_FILE,
            column_name=COLUMN_NAME,
            reference_csv=SEARCH_ROOT,
            output_column=OUTPUT_COLUMN,
            exact_match=EXACT_MATCH
        )
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 如果你想直接运行，请修改main()中的参数
    # 或者在命令行中导入使用：
    # >>> from folder_finder import process_csv
    # >>> process_csv("data.csv", "folder_column", "/search/path")

    main()