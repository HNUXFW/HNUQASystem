from typing import List, Dict
import requests
class Generator:
    def __init__(self):
        # 本地模型路径配置，使用ollama进行配置
        self.ollama_url = "http://localhost:11434/api/chat"  # Ollama默认API地址
        self.model_name = "deepseek-r1:7b"  # 确保已通过ollama pull deepseek-r1:7b下载该模型
    def generate_response(self, query: str, relevant_docs: List[Dict]) -> str:
        """生成回答（Ollama API版本）"""
        # 构建系统提示词
        system_prompt = """你是一个校园智能问答助手。请基于提供的相关文档信息，回答用户的问题。
             注意事项：
             1. 只回答与校园相关的问题
             2. 如果不确定或没有相关信息，请提示用户咨询人工服务
             3. 保持回答准确、简洁、专业
             4. 不要编造信息或做出未经证实的承诺"""

        # 添加获取的上下文
        context = "\n\n".join([f"文档片段 {i + 1}:\n{doc['text']}"
                               for i, doc in enumerate(relevant_docs)])

        # 构建Ollama所需的对话格式
        messages = [
            {
                "role": "system",
                "content": system_prompt,
                "images": []
            },
            {
                "role": "user",
                "content": f"基于以下文档信息回答问题：\n{context}\n\n用户问题：{query}",
                "images": []
            }
        ]

        try:
            # 调用Ollama API
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 500  # 相当于max_tokens
                    },
                    "stream": False  # 非流式响应
                },
                timeout=120  # 超时设置为2分钟
            )

            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                error_msg = f"API请求失败，状态码：{response.status_code}"
                if response.text:
                    error_msg += f"，响应内容：{response.text[:200]}"
                return f"抱歉，服务暂时不可用。{error_msg}"

        except Exception as e:
            return f"抱歉，生成回答时出现错误。建议您咨询人工服务。错误信息：{str(e)}"