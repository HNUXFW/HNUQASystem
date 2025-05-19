from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from utils import DocumentProcessor,Generator
from typing import List, Dict

@asynccontextmanager
async def lifespan(app:FastAPI):
    try:
        print("开始加载文档和索引...")
        # 确保必要的目录存在
        settings.docs_dir.mkdir(parents=True, exist_ok=True)
        settings.vector_dir.mkdir(parents=True, exist_ok=True)

        # 尝试加载现有索引，如果不存在则构建新索引
        index_path = settings.vector_dir / "faiss_index"
        print(f"现有索引: {index_path}")
        if index_path.exists():
            doc_processor.load_index(index_path)
        else:
            doc_processor.process_documents(settings.docs_dir)
            doc_processor.build_index()
            doc_processor.save_index(index_path)
        yield
        print("关闭文档和索引...")
    except Exception as e:
        print(f"启动时出错: {str(e)}")
        raise

app = FastAPI(
    title=settings.project_name,
    version=settings.version,
    description="校园智能问答系统API文档",
    docs_url="/docs",   # Swagger UI 的URL
    redoc_url="/redoc" , # ReDoc 的URL
    lifespan=lifespan
)

# 配置CORS 避免跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#配置模板引擎，用于渲染主页
templates = Jinja2Templates(directory="templates")

# 初始化文档处理器和生成器
doc_processor = DocumentProcessor()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """渲染主页"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": settings.project_name}
    )



@app.post("/api/v1/qa")
async def qa_endpoint(
    request: Dict = Body(...)
):
    """问答接口
    
    Args:
        request (Dict): 包含查询和历史记录的请求体
            - query (str): 用户的问题
            - history (List[Dict]): 对话历史记录
                - content (str): 消息内容
                - role (str): 角色（user/assistant）
        
    Returns:
        dict: 包含答案和相关文档来源的响应
    """
    try:
        query = request.get("query", "")
        history = request.get("history", [])
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
            
        # 检索相关文档
        relevant_docs = doc_processor.search(query)
        
        # 生成回答
        response = Generator.generate_response(query, relevant_docs, history)
        sources = []
        for doc in relevant_docs:
            if doc["source"] not in sources:
                sources.append(doc["source"])
        
        return {
            "answer": response,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

