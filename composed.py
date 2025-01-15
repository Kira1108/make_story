from identity_verification_agent import IdentityVerificationAgent
from business_agent import BusinessAgent

class CompoundAgent:
    
    def __init__(self):
        self.identity_verification_agent = IdentityVerificationAgent()
        self.business_agent = BusinessAgent(temperature=0)
        self.main_agent = self.identity_verification_agent
        
    def chat(self, text:str):
        response = self.main_agent.chat(text)
        if self.main_agent == self.identity_verification_agent:
            if self.main_agent.identity_verified:
                self.main_agent = self.business_agent
                self.business_agent.continue_from(self.identity_verification_agent)
        return response