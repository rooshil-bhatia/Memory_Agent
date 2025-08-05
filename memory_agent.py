import os
import time
from groq import Groq, RateLimitError
from mem0 import Memory
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import json


load_dotenv()

class MemoryAgent:
    
    def __init__(self, user_id: str = "default-user"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = "gemma2-9b-it" 
        
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        
        mem0_config = {
            "llm": {"provider": "groq", "config": {"model": self.model, "api_key": os.environ.get("GROQ_API_KEY")}},
            "embedder": {"provider": "langchain", "config": {"model": embedding_model}},
            "vector_store": {"provider": "qdrant", "config": {"on_disk": False, "embedding_model_dims": 384}}
        }
        
        self.memory = Memory.from_config(mem0_config)
        self.user_id = user_id
        
        print(f"Memory Agent initialized for user: {self.user_id} with model {self.model}.")

    def _safe_parse_memories(self, memory_data) -> list:
        if not memory_data: return []
        
        memory_list = memory_data

        if isinstance(memory_data, dict) and 'results' in memory_data:
            memory_list = memory_data['results']
        
        parsed_memories = []
        for item in memory_list:
            if isinstance(item, dict):
                parsed_memories.append(item.get('memory', str(item)))
            else:
                parsed_memories.append(str(item))
        return parsed_memories

    def list_all_memories(self) -> str:
        try:
            response = self.memory.get_all(user_id=self.user_id)
            parsed_memories = self._safe_parse_memories(response)
            
            if not parsed_memories: return "You have no memories stored."

            formatted_memories = "\n".join(f"- {text}" for text in parsed_memories)
            return f"Here are your stored memories:\n{formatted_memories}"
        except Exception as e:
            return f"Could not retrieve memories: {e}"

    def analyze_and_manage_memory(self, user_input: str):
        
        all_memories = self._safe_parse_memories(self.memory.get_all(user_id=self.user_id))
        
        system_prompt = f"""
You are an intelligent memory controller. Analyze the "LATEST USER STATEMENT" and decide if it introduces a new fact or updates an existing topic.

Current Memories:
{json.dumps(all_memories, indent=2)}

Your task is to respond with a JSON object.
- If the statement is a new fact, use the key "new_fact".
- If the statement updates an existing memory, use the key "updated_fact" and provide the general "topic" being updated.
- If no memory action is needed, return an empty JSON object.
**You must respond in JSON format.**

Example 1: User says "I live in New York." -> {{"new_fact": "User lives in New York."}}
Example 2: User says "I don't play football anymore, I play basketball now." -> {{"topic": "user's current sport", "updated_fact": "User now plays basketball."}}
Example 3: User says "My favorite team is now Chelsea." -> {{"topic": "user's favorite football team", "updated_fact": "User's favorite team is Chelsea."}}
Example 4: User says "What's the weather like?" -> {{}}

LATEST USER STATEMENT: "{user_input}"
"""

        messages = [{"role": "system", "content": system_prompt}]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, response_format={"type": "json_object"}
            )
            analysis = json.loads(response.choices[0].message.content)
            print(f"Memory Analysis: {analysis}")

            topic = analysis.get("topic")
            new_fact = analysis.get("new_fact") or analysis.get("updated_fact")

            if topic:
                memories_to_delete = self.memory.search(query=topic, user_id=self.user_id, limit=5)
                for mem in memories_to_delete:
                    if isinstance(mem, dict) and 'id' in mem:
                        self.memory.delete(id=mem['id'])
                        print(f"Memory DELETED on topic '{topic}': {mem.get('memory')}")

            if new_fact:
                
                memory_to_add = [{"role": "user", "content": new_fact}]
                self.memory.add(messages=memory_to_add, user_id=self.user_id)
                print(f"Memory CREATED/UPDATED: {new_fact}")

        except Exception as e:
            print(f"Could not analyze or manage memory. Error Type: {type(e).__name__}, Details: {e}")

    def process_message(self, user_input: str, conversation_history: list) -> str:
        
        self.analyze_and_manage_memory(user_input)
        
        relevant_memories = self._safe_parse_memories(self.memory.search(query=user_input, user_id=self.user_id))
        memories_str = "\\n".join(f"- {mem}" for mem in relevant_memories)
        
        system_prompt = f"""
You are a friendly, insightful, and conversational AI assistant with a perfect memory.
The memories below are a log of facts about the user. Newer facts can contradict and override older ones.
Your primary job is to synthesize these facts and respond based on the most up-to-date information.
Do NOT just state the memories back to the user; weave your understanding from them into a natural conversation.

**MEMORIES:**
{memories_str}
"""
        messages = conversation_history + [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
        
        try:
            response = self.client.chat.completions.create(model=self.model, messages=messages)
            assistant_response = response.choices[0].message.content
            
            
            self.memory.add(messages=messages, user_id=self.user_id)
            
            return assistant_response
        except Exception as e:
            return f"An error occurred while generating a response: {e}"

    def start_chat(self):
        
        conversation_history = []
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == 'exit':
                print("AI: Goodbye!")
                break
            if user_input.lower() == 'list memories':
                print(f"AI:\n{self.list_all_memories()}")
                continue
            
            ai_response = self.process_message(user_input, conversation_history)
            print(f"AI: {ai_response}")

            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": ai_response})

if __name__ == "__main__":
    agent = MemoryAgent(user_id="user1")
    agent.start_chat()
