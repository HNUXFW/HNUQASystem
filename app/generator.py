from typing import List, Dict
from app.config import settings

class Generator:
    def __init__(self):
        #TODO 这里准备替换为本地的模型，如Deepseek

            
    def generate_response(self, query: str, relevant_docs: List[Dict]) -> str:
        """生成回答"""
        # 构建系统提示词
        system_prompt = """你是一个校园智能问答助手。请基于提供的相关文档信息，回答用户的问题。
        注意事项：
        1. 只回答与校园相关的问题
        2. 如果不确定或没有相关信息，请提示用户咨询人工服务
        3. 保持回答准确、简洁、专业
        4. 不要编造信息或做出未经证实的承诺"""
        
        # 构建上下文
        context = "\n\n".join([f"文档片段 {i+1}:\n{doc['text']}" 
                              for i, doc in enumerate(relevant_docs)])
        
        # 构建用户提示词
        user_prompt = f"""基于以下文档信息回答问题：{context}
        用户问题：{query}
        请提供准确的回答："""
        
        try:
            # 调用OpenAI API生成回答
            # response = openai.ChatCompletion.create(
            #     model="gpt-3.5-turbo",
            #     messages=[
            #         {"role": "system", "content": system_prompt},
            #         {"role": "user", "content": user_prompt}
            #     ],
            #     temperature=0.7,
            #     max_tokens=500
            # )
            # return response.choices[0].message.content
            
        except Exception as e:
            return f"抱歉，生成回答时出现错误。建议您咨询人工服务。错误信息：{str(e)}" 