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
- 讯飞星火 API: 答案生成
- TailwindCSS: 前端样式

## 安装说明

1. 克隆项目：
```bash
git clone [项目地址]
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
之后将.env中的EMBEDDING_MODEL替换为模型路径。
此外，还需要解决tesseract问题。具体参考https://developer.aliyun.com/article/1528694

## 使用说明

1. 准备文档：
   - 在 `data/docs` 目录下放置文本文档（.txt/.docx/.pdf 格式）
   - 文档应包含校园相关的 FAQ、规章制度等内容

2. 启动服务：
```bash
python main.py
```

3. 访问系统：
   - 打开浏览器访问 http://localhost:8000
   - 在网页中输入问题
   - 系统会返回相关答案和参考来源，注意由于使用的是免费的API，结果可能会不准确

## 目录结构

```
.
├── app/
│   ├── document_processor.py  # 文档处理模块
│   ├── generator.py       # 答案生成模块
├── data/
│   ├── docs/             # 文档存储目录
│   └── vectors/          # 向量索引存储目录
│—— config/               # 配置文件
├── templates/            # HTML模板
|—— main.py               # 主程序
├── app.py                # FastAPI 应用
├── README.md            # 项目说明
├── environment.yml      # 虚拟环境配置
└── .env                 # 环境变量
```
## 最终结果
![image](https://github.com/user-attachments/assets/c9444f6e-cc9d-41a9-8eb8-11370608ebaa)


## 注意事项
- 首次启动时会自动构建向量索引，并且由于添加了对于图片的识别，所以对于图片很多的文档，构建索引时间会比较长。
- 建议定期更新文档和重建索引以保持信息最新
- 由于采用的chunk的划分，搜索的信息可能出现割裂，导致结果不准确。此外，从pdf中识别信息，还存在一些换行的问题


