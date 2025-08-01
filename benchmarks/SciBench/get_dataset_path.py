import os

def get_custom_files(directory, extensions,included_names = None, max_depth=None):
    """
    获取指定目录下所有具有特定扩展名的文件路径，支持自定义搜索深度和相对路径

    参数:
        directory (str): 要搜索的目录路径（支持相对路径）
        extensions (list): 包含目标扩展名的列表（如 ['json', 'csv']）
        max_depth (int): 最大搜索深度（None 表示不限制深度）
        included_names (str): 目标文件名需要包含的特征

    返回:
        list: 包含所有匹配文件完整路径的列表

    异常:
        ValueError: 当指定目录不存在时抛出
    """
    # 检查目录是否存在
    if not os.path.isdir(directory):
        raise ValueError(f"目录 '{directory}' 不存在")

    directory = os.path.abspath(directory)  # 转换为绝对路径
    base_level = directory.count(os.sep)  # 计算基础目录的层级
    
    processed_exts = ['.' + ext.strip('.').lower() for ext in extensions]
    
    matched_files = []
    
    for root, dirs, files in os.walk(directory):
        # 计算当前目录相对于基础目录的深度
        current_level = root.count(os.sep)
        current_depth = current_level - base_level
        
        if max_depth is not None and current_depth > max_depth:
            dirs[:] = []  # 阻止 os.walk 进入更深的子目录
            continue
        
        for filename in files:
            # 获取文件扩展名并转换为小写
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in processed_exts:
                full_path = os.path.join(root, filename)
                matched_files.append(full_path)
    
    if included_names is not None:
        if isinstance(included_names, str):
            # 如果是单个字符串
            matched_files = [ff for ff in matched_files if included_names in ff]
        else:
            # 如果是字符串列表，文件路径需要包含列表中所有字符串
            print('选中')
            matched_files = [ff for ff in matched_files if all(name in ff for name in included_names)]
    
    return matched_files

    
if __name__ == "__main__":
    print('get_dataset_path')
    files = get_custom_files(
        directory = './HighQuality/OlympicArena',  # 相对路径和绝对路径都行
        extensions = ['.parquet'] ,        # 要搜索的文件类型 ['csv', 'json', 'xlsx']
        included_names = ['OlympicArena','Geography'],  # 其他限制要求，例如数据集和科目，None就是所有
        max_depth=1           # 搜索深度，0是只搜索当前目录
    )  
    
    