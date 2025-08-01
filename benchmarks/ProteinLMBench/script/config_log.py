import logging
import os
from typing import Optional

def setup_logger(name: Optional[str] = None, log_file: str = "app.log") -> logging.Logger:
    """
    配置并返回一个日志器实例
    :param name: 日志器名称（建议使用项目名，如 "myproject"）
    :param log_file: 日志文件路径
    :return: 配置好的 logger 实例
    """
    # 若不指定名称，返回根日志器；否则返回指定名称的日志器
    logger = logging.getLogger(name)
    
    # 避免重复添加 handlers（多次调用时的关键）
    if logger.handlers:
        return logger
    
    # 设置日志级别（DEBUG < INFO < WARNING < ERROR < CRITICAL）
    logger.setLevel(logging.INFO)
    
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
      # 定义格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # 仅添加文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    

    return logger