from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 基础配置
    PROJECT_NAME: str = "校园智能问答系统"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # 获取项目根目录
    BASE_DIR: Path = Path(__file__).parent.parent.absolute()
    
    # 文档相关配置
    DOCS_DIR: Path = BASE_DIR / "data" / "docs"
    VECTOR_DIR: Path = BASE_DIR / "data" / "vectors"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # 模型相关配置
    EMBEDDING_MODEL: str = "D:\CodeProject\paraphrase-multilingual-MiniLM-L12-v2"
    TOP_K: int = 3
    
    # OpenAI配置（如果使用）
    OPENAI_API_KEY: str = ""
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 