from typing import List, Dict
import requests

from config import settings


class Generator:
    @staticmethod
    def format_history(history: List[Dict]) -> str:
        """格式化对话历史记录"""
        if not history:
            return ""
            
        formatted_history = "\n对话历史：\n"
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            formatted_history += f"{role}：{msg['content']}\n"
        return formatted_history

    @staticmethod
    def generate_response(query: str, relevant_docs: List[Dict], history: List[Dict] = None) -> str:
        """生成回答
        
        Args:
            query (str): 用户的问题
            relevant_docs (List[Dict]): 相关文档列表
            history (List[Dict], optional): 对话历史记录
            
        Returns:
            str: 生成的回答
        """
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

        # 添加对话历史
        history_text = Generator.format_history(history) if history else ""
        
        # 构建完整的提示词
        prompt = f"""请基于以下信息回答用户的问题。如果无法从提供的信息中找到答案，请明确说明。
                相关文档：{context}
                历史信息：{history_text}
                当前问题：{query}
                请用简洁、专业的语言回答问题，并确保回答的准确性。如果需要引用具体内容，请说明来源。"""

        # 构建API所需的对话格式
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]

        try:

            response = requests.post(
                settings.llm_url,
                json={
                    "model": settings.model_name,
                    "messages": messages,
                    "stream": False  # 非流式响应
                },
                headers={
                    "Authorization": f"Bearer {settings.api_password}"
                },
                timeout=120  # 超时设置为2分钟
            )

            if response.status_code == 200:
                print(response.json())
                return response.json()["choices"][0]["message"]["content"]
            else:
                error_msg = f"API请求失败，状态码：{response.status_code}"
                if response.text:
                    error_msg += f"，响应内容：{response.text[:200]}"
                return f"抱歉，服务暂时不可用。{error_msg}"

        except Exception as e:
            return f"抱歉，生成回答时出现错误。建议您咨询人工服务。错误信息：{str(e)}"