from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from pathlib import Path
import uvicorn

from app.config import settings
from app.document_processor import DocumentProcessor
from app.generator import Generator

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 初始化文档处理器和生成器
doc_processor = DocumentProcessor()
generator = Generator()

@app.on_event("startup")
async def startup_event():
    """启动时加载文档和索引"""
    try:
        # 确保必要的目录存在
        settings.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        settings.VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        
        # 处理文档
        doc_processor.process_documents(settings.DOCS_DIR)
        
        # 尝试加载现有索引，如果不存在则构建新索引
        index_path = settings.VECTOR_DIR / "faiss_index"
        if index_path.exists():
            doc_processor.load_index(index_path)
        else:
            doc_processor.build_index()
            doc_processor.save_index(index_path)
            
    except Exception as e:
        print(f"启动时出错: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """渲染主页"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": settings.PROJECT_NAME}
    )

@app.post("/api/v1/qa")
async def qa_endpoint(query: str):
    """问答接口"""
    try:
        # 检索相关文档
        relevant_docs = doc_processor.search(query)
        
        # 生成回答
        response = generator.generate_response(query, relevant_docs)
        
        return {
            "answer": response,
            "sources": [
                {
                    "text": doc["text"],
                    "source": doc["source"]
                }
                for doc in relevant_docs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 