from llama_index.core.agent import FunctionCallingAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from llm_config import LLM_NAME

FAKE_ORDER_INFO = """
| 贷款批准日期       | 贷款还款日期       | 金额    | 车系     | 利率     | 借款人姓名   | 贷款状态   |
|--------------------|--------------------|---------|----------|----------|--------------|------------|
| 2023-01-15         | 2026-01-15         | $20,000 | Tesla Model Y    | 3.5%     | 约翰·多伊    | 已批准     |
| 2023-02-20         | 2026-02-20         | $25,000 | 宝马5系    | 4.0%     | 约翰·多伊    | 已批准     |
| 2023-03-10         | 2026-03-10         | $30,000 | 保时捷 911    | 3.8%     | 约翰·多伊  | 已批准     |
| 2023-04-05         | 2026-04-05         | $22,000 | 丰田凯美瑞    | 3.9%     | 约翰·多伊  | 已批准     |
| 2023-05-18         | 2026-05-18         | $28,000 | 奔驰S级    | 4.2%     | 约翰·多伊 | 已批准     |
""".strip()


def get_info(id_number:str) -> str:
    """Get the user's loan order information by their ID number."""
    print("[Tool calling] Querying loan info for user with ID number:", id_number)
    return FAKE_ORDER_INFO

SYSTEM_PROMPT="""
# Role: You are a customer service representative for a car sales/loan company, communicating with users over the phone to help them resolve issues.

# Style: Respond in a professional and conversational tone, as if speaking on the phone.

# Tasks:
1. To answer user questions, you need to use tools AS MUCH AS POSSIBLE.
2. Anything related to the user's orders loans, or related to the business, MAKE SURE TO REFER TO AT LEAST ONE OF THE TOOLS.

# Constraints:
1. Keep responses concise and to the point.
2. Use interactive responses. Do not output too much information at once. (Minimize the number of words in each response)
3. Uset the same language as the user. If the user speaks English, respond in English. If the user speaks Chinese, respond in Chinese.

""".strip()

order_info_retrieval_tool = FunctionTool.from_defaults(
    get_info,
)

class BusinessAgent:
    
    def __init__(self, temperature = 0):
        self.agent = FunctionCallingAgent.from_tools(
            tools = [order_info_retrieval_tool], 
            system_prompt=SYSTEM_PROMPT, 
            llm = Ollama(model=LLM_NAME,temperature=temperature))
        
    def continue_from(self, agent):
        self.agent.memory.chat_store.store['chat_history'] = agent.chat_messages
        return self
        
    @property
    def chat_messages(self):
        messages = self.agent.memory.chat_store.store['chat_history']
        return messages
        
    def chat(self, query:str) -> str:
        response = self.agent.chat(query)
        return response




