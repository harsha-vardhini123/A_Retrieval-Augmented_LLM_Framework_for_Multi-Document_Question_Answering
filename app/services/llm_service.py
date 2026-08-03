import json
import io
import re
from .storage_service import S3Storage
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from config import config

class LLMService:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.storage = S3Storage()

    def _get_chat_history_from_s3(self, session_id):
        try:
            obj = self.storage.get_object(f"chat_history/{session_id}.json")
            content = obj.read().decode("utf-8")
            return json.loads(content)
        except Exception:
            return []

    def _save_chat_history_to_s3(self, session_id, history):
        try:
            data = json.dumps(history)
            self.storage.upload_bytes(io.BytesIO(data.encode("utf-8")), f"chat_history/{session_id}.json")
        except Exception as e:
            print(f"Failed to save chat history: {e}")

    def _post_process_response(self, response):
        """Clean and format the response"""
        # Remove simple markdown formatting if the model still uses it
        response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)
        response = re.sub(r'\*(.*?)\*', r'\1', response)
        response = re.sub(r'#+\s', '', response)
        
        # Clean up multiple newlines
        response = re.sub(r'\n\s*\n', ' ', response)
        
        # Ensure it ends with proper punctuation if it doesn't already
        response = response.strip()
        if response and response[-1] not in '.!?':
            response += '.'
            
        return response

    def get_response(self, query, session_id, search_type="similarity"):
        try:
            
            use_history = not session_id.startswith("eval")
            
            history = self._get_chat_history_from_s3(session_id) if use_history else []

            # Setup memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

            # Inject previous Q&A only if using history
            if use_history:
                for item in history:
                    memory.chat_memory.add_user_message(item["question"])
                    memory.chat_memory.add_ai_message(item["answer"])

            # Setup LLM with lower temperature for more consistent responses
            """
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.1,  # Reduced for more consistent evaluation
                google_api_key=config.GEMINI_API_KEY
            )
            """

            # Use a more powerful, accurate model with zero temperature for factual retrieval
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.0
            )

            
            qa_prompt = PromptTemplate(
                template="""You are a precise and expert AI assistant. Answer questions accurately based ONLY on the Context provided.

Guidelines:
- If the user is just saying a simple greeting (like "Hi", "Hello", "How are you?"), you MUST respond with a warm, friendly greeting and ask how you can help them with the knowledge base. Ignore the context rule for simple greetings.
- For all other questions, provide highly factual, direct answers.
- Base your answer strictly on the provided Context and Chat History. Do NOT hallucinate or use outside knowledge.
- If the Context does not contain the answer to their question, explicitly state: "The provided context does not contain the answer."
- Use 2-3 clear, complete sentences maximum.
- Focus strictly on key information.
- Do not use bullet points, markdown, or lists.

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer:""",
                input_variables=["context", "chat_history", "question"]
            )

            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vector_store.as_retriever(search_type=search_type, search_kwargs={"k": 6}),
                memory=memory,
                combine_docs_chain_kwargs={"prompt": qa_prompt},
                return_source_documents=False,
                verbose=False
            )

            result = chain({"question": query})
            answer = result['answer']

            # Apply post-processing to clean up the response
            answer = self._post_process_response(answer)

            # Save history only for non-evaluation sessions
            if use_history:
                history.append({
                    "question": query,
                    "answer": answer
                })
                self._save_chat_history_to_s3(session_id, history)

            return answer
            
        except Exception as e:
            print(f"Error in get_response: {e}")
            return "An error occurred processing your request."