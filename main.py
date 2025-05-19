import uvicorn
#TODO 关于切分的问题。识别出来的chunk内容总是带有\n使得语义不连贯，大模型理解可能出错。
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)