from llama_index.core.agent import FunctionCallingAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage
import json
from llm_config import LLM_NAME

# id_verification_tool.metadata.description
# "id_verify(id_number: str) -> int\nVerify the user's ID number." 把函数的签名作为system prompt的一部分


def id_verify(id_number:str) -> dict:
    """Verify the user's ID number."""
    print("[Tool calling] Verifying Identity: ", id_number)
    return {"verified": True, "id_number": id_number}

id_verification_tool = FunctionTool.from_defaults(
    id_verify,
)

SYSTEM_PROMPT="""
# Role: You are a customer service representative for a car sales/loan company, communicating with users over the phone to help them resolve issues.

# Style: Respond in a professional and conversational tone, as if speaking on the phone.

# Tasks:
1. If the user mentions issues related to orders or loans, prompt the user to enter their ID number. Only after verifying their identity can you continue to provide consultation on order or loan issues.
2. Note: If you need to prompt the user to enter their ID information, ask for the full 18-digit ID number. Partial ID numbers (e.g., last few digits) are not acceptable for querying user information.
3. Do not proactively introduce your capabilities. Only respond to questions asked by the user.
4. If the user does not ask about order-related issues, do not ask for their ID number, unexpectedly asking for personal information is RUDE and is strictly forbiddened.

# Constraints:
1. Keep responses concise and to the point.
2. Use interactive responses. Do not output too much information at once. (Minimize the number of words in each response)
3. Uset the same language as the user. If the user speaks English, respond in English. If the user speaks Chinese, respond in Chinese.
4. If the user provides and ID number, you must use tools to verify the ID number before providing any further information.

Note: The user questions are converted by ASR, so you should handle special case like double 5 -> 55, three 8 -> 888, and other interupted numbers.

""".strip()

class IdentityVerificationAgent:
    
    def __init__(self, temperature = 0):
        self.agent = FunctionCallingAgent.from_tools(
            tools = [id_verification_tool], 
            system_prompt=SYSTEM_PROMPT, 
            llm = Ollama(model=LLM_NAME,temperature=temperature))
        
        self.identity_verified = False
        self.identity_number = None
    
    @property
    def chat_messages(self):
        messages = self.agent.memory.chat_store.store['chat_history']
        return messages
    
    def fix_messages(self):
        messages = self.chat_messages
        if self.identity_verified:
            verify_message = f"好的，你的身份已验证通过，尾号{self.identity_number[-4:]}。"
            messages = messages[:-1] + [ChatMessage(role = "assistant", content = verify_message )]
        self.agent.memory.chat_store.store['chat_history'] = messages
        
    def chat(self, query:str) -> str:
        response = self.agent.chat(query)
        output = response.response
        
        if len(response.sources) > 0:
            tool_info = eval(response.sources[0].content)
            if tool_info['verified']:
                self.identity_verified = True
                self.identity_number = tool_info['id_number']
                self.fix_messages()
                output = f"好的，你的身份已验证通过，尾号{self.identity_number[-4:]}。"
        return output
