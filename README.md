# 校园智能问答系统

基于 RAG (检索增强生成) 的校园问答系统，可以智能回答校园相关问题。

## 功能特点

- 基于文档的智能问答
- 语义检索匹配相关文档
- 使用大语言模型生成准确回答
- 现代化的 Web 界面
- 支持显示答案来源

## 技术栈

- FastAPI: Web 框架
- Sentence-Transformers: 文本向量化
- FAISS: 向量检索
- OpenAI API: 答案生成
- TailwindCSS: 前端样式

## 安装说明

1. 克隆项目：
```bash
git clone [项目地址]
cd [项目目录]
```


2.安装依赖：
为了避免库之间冲突，最好使用conda进行管理：
```bash
conda env create -f environment.yml
```
paraphrase-multilingual-MiniLM-L12-v2模型，采用国内镜像安装：
```bash
git clone https://gitcode.com/hf_mirrors/ai-gitcode/paraphrase-multilingual-MiniLM-L12-v2
```
zh_core_web_sm模型下载命令：
```bash
python -m spacy download zh_core_web_sm
```
此外，还需要解决tesseract 问题。具体参考https://developer.aliyun.com/article/1528694

## 使用说明

1. 准备文档：
   - 在 `data/docs` 目录下放置文本文档（.txt/.doc/.docx/.pdf 格式）
   - 文档应包含校园相关的 FAQ、规章制度等内容

2. 启动服务：
```bash
python main.py
```

3. 访问系统：
   - 打开浏览器访问 http://localhost:8000/docs
   - 在相关API中输入问题
   - 系统会返回相关答案和参考来源

## 目录结构

```
.
├── app/
│   ├── app.py             # FastAPI 应用
│   ├── config.py          # 配置文件
│   ├── document_processor.py  # 文档处理模块
│   ├── generator.py       # 答案生成模块
│   └── main.py           # 主程序
├── data/
│   ├── docs/             # 文档存储目录
│   └── vectors/          # 向量索引存储目录
├── static/               # 静态文件
├── templates/            # HTML 模板
├── requirements.txt      # 项目依赖
└── README.md            # 项目说明
```

## 注意事项
- 确保文档格式正确（UTF-8 编码的 .txt 文件）
- 首次启动时会自动构建向量索引，可能需要一些时间
- 建议定期更新文档和重建索引以保持信息最新

## 贡献指南
欢迎提交 Issue 和 Pull Request 来改进系统。

