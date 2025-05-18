from pathlib import Path
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # 基础配置
    project_name: str
    version: str
    api_v1_str: str

    # 文档相关配置
    docs_dir: Path
    vector_dir: Path
    chunk_size: int
    chunk_overlap: int

    # 模型相关配置
    embedding_model: str
    top_k: int

    #大模型相关配置
    llm_url: str
    api_password: str
    model_name: str

    class Config:
        case_sensitive = False# 大小写不敏感
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        env_file_encoding = "utf-8"


