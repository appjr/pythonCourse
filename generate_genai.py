#!/usr/bin/env python3
import json

def create_notebook(title, sections):
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': [f'# {title}\n\n---\n']}]
    for section in sections:
        if 'markdown' in section:
            cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': section['markdown']})
        if 'code' in section:
            cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': section['code']})
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.10.0'}}, 'nbformat': 4, 'nbformat_minor': 4}

notebooks = {
    '06-generative-ai/01-intro-to-genai.ipynb': {
        'title': '🌟 GenAI: Introduction',
        'sections': [
            {'markdown': ['## 🤖 What is Generative AI?\n\nGenerative AI creates new content: text, images, code, etc.\n\n**Applications:**\n- ChatGPT, Claude\n- DALL-E, Midjourney\n- GitHub Copilot\n']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ GenAI creates content\n✅ Based on transformers\n✅ Revolutionary technology\n']}
        ]
    },
    '06-generative-ai/02-transformers.ipynb': {
        'title': '🌟 GenAI: Transformers',
        'sections': [
            {'markdown': ['## 🔄 Transformer Architecture\n\n**Key Components:**\n- Self-attention mechanism\n- Multi-head attention\n- Feed-forward networks\n- Positional encoding\n']},
            {'code': ['# pip install transformers\nfrom transformers import pipeline\n\n# Text classification\nclassifier = pipeline("sentiment-analysis")\nresult = classifier("I love this course!")\nprint(result)']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ Attention is all you need\n✅ Transformers revolutionized NLP\n✅ Foundation of modern AI\n']}
        ]
    },
    '06-generative-ai/03-gans.ipynb': {
        'title': '🌟 GenAI: GANs',
        'sections': [
            {'markdown': ['## 🎨 Generative Adversarial Networks\n\n**Components:**\n- Generator: Creates fake data\n- Discriminator: Detects fake vs real\n- Adversarial training\n']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ GANs generate realistic content\n✅ Generator vs Discriminator\n✅ Used for images, video, audio\n']}
        ]
    },
    '06-generative-ai/04-llm-fundamentals.ipynb': {
        'title': '🌟 GenAI: LLM Fundamentals',
        'sections': [
            {'markdown': ['## 🧠 Large Language Models\n\n**Key Concepts:**\n- Next token prediction\n- Context window\n- Temperature sampling\n- GPT architecture\n']},
            {'code': ['# Using OpenAI API\nimport openai\n\n# Set your API key\n# openai.api_key = "your-key"\n\n# completion = openai.ChatCompletion.create(\n#     model="gpt-3.5-turbo",\n#     messages=[{"role": "user", "content": "Hello!"}]\n# )\n# print(completion.choices[0].message.content)']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ LLMs predict next tokens\n✅ Trained on massive text data\n✅ Few-shot learning capability\n']}
        ]
    },
    '06-generative-ai/05-prompt-engineering.ipynb': {
        'title': '🌟 GenAI: Prompt Engineering',
        'sections': [
            {'markdown': ['## 📝 Prompt Engineering\n\n**Techniques:**\n- Zero-shot prompting\n- Few-shot examples\n- Chain-of-thought\n- Role-based prompts\n- System messages\n']},
            {'code': ['# Example prompts\nprompts = {\n    "zero_shot": "Translate to French: Hello",\n    "few_shot": """Translate to French:\nHello -> Bonjour\nGoodbye -> Au revoir\nThank you -> """,\n    "cot": "Think step by step: What is 25 * 4?",\n    "role": "You are a Python expert. Explain list comprehensions."\n}\n\nfor name, prompt in prompts.items():\n    print(f"{name}:\\n{prompt}\\n")']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ Prompts control LLM behavior\n✅ Clear, specific prompts work best\n✅ Iterate and refine\n']}
        ]
    },
    '06-generative-ai/06-rag-retrieval-augmented-generation.ipynb': {
        'title': '🌟 GenAI: RAG (Retrieval-Augmented Generation)',
        'sections': [
            {'markdown': ['## 📚 What is RAG?\n\nRAG enhances LLMs with external knowledge.\n\n**Components:**\n- Vector database (ChromaDB, Pinecone, FAISS)\n- Embeddings (semantic search)\n- Document retrieval\n- Context injection\n']},
            {'code': ['# pip install langchain chromadb openai\nfrom langchain.embeddings import OpenAIEmbeddings\nfrom langchain.vectorstores import Chroma\nfrom langchain.text_splitter import CharacterTextSplitter\n\n# Sample documents\ndocs = [\n    "Python is a programming language.",\n    "RAG retrieves relevant documents.",\n    "Vector databases store embeddings."\n]\n\n# text_splitter = CharacterTextSplitter(chunk_size=100)\n# chunks = text_splitter.create_documents(docs)\nprint("RAG pipeline setup complete!")']},
            {'markdown': ['## 🎯 Building RAG Systems\n\n**Steps:**\n1. Chunk documents\n2. Create embeddings\n3. Store in vector DB\n4. Retrieve relevant docs\n5. Inject into LLM prompt\n']},
            {'code': ['# Simple RAG query\ndef rag_query(question, docs):\n    # 1. Retrieve relevant docs\n    relevant_docs = [doc for doc in docs if any(word in doc.lower() for word in question.lower().split())]\n    # 2. Create context\n    context = "\\n".join(relevant_docs)\n    # 3. Create prompt\n    prompt = f"Context: {context}\\n\\nQuestion: {question}"\n    return prompt\n\nquestion = "What is Python?"\nresult = rag_query(question, ["Python is a language.", "RAG enhances LLMs."])\nprint(result)']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ RAG adds external knowledge to LLMs\n✅ Vector databases enable semantic search\n✅ Perfect for company knowledge bases\n✅ More accurate than fine-tuning for many use cases\n']}
        ]
    },
    '06-generative-ai/07-ai-agents.ipynb': {
        'title': '🌟 GenAI: AI Agents',
        'sections': [
            {'markdown': ['## 🤖 What are AI Agents?\n\nAutonomous systems that can:\n- Use tools\n- Make decisions\n- Execute multi-step tasks\n- Learn from feedback\n\n**Agent Architectures:**\n- ReAct (Reasoning + Acting)\n- Plan-and-Execute\n- Reflexion (self-reflection)\n']},
            {'code': ['# pip install langchain\nfrom langchain.agents import AgentType, initialize_agent, load_tools\nfrom langchain.llms import OpenAI\n\n# Example agent setup\n# llm = OpenAI(temperature=0)\n# tools = load_tools(["python_repl"], llm=llm)\n# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)\n\nprint("Agent framework setup!")']},
            {'markdown': ['## 🛠️ Tool Use & Function Calling\n\n**Capabilities:**\n- Calculator\n- Web search\n- Code execution\n- Database queries\n- API calls\n']},
            {'code': ['# Simple agent example\nclass SimpleAgent:\n    def __init__(self):\n        self.tools = {\n            "calculator": lambda x: eval(x),\n            "uppercase": lambda x: x.upper()\n        }\n    \n    def execute(self, tool_name, input_data):\n        if tool_name in self.tools:\n            return self.tools[tool_name](input_data)\n        return "Tool not found"\n\nagent = SimpleAgent()\nprint(agent.execute("calculator", "5 * 10"))\nprint(agent.execute("uppercase", "hello"))']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ Agents are autonomous AI systems\n✅ Can use tools and APIs\n✅ Multi-step reasoning\n✅ Foundation of AI automation\n']}
        ]
    },
    '06-generative-ai/08-mcp-model-context-protocol.ipynb': {
        'title': '🌟 GenAI: MCP (Model Context Protocol)',
        'sections': [
            {'markdown': ['## 🔌 What is MCP?\n\nModel Context Protocol is a standard for connecting AI models to data sources.\n\n**Components:**\n- MCP Server: Provides tools/resources\n- MCP Client: Uses tools/resources\n- Protocol: Standardized communication\n']},
            {'markdown': ['## 🏗️ MCP Architecture\n\n**Server Types:**\n- Tools: Functions the model can call\n- Resources: Data the model can access\n- Prompts: Pre-defined prompt templates\n']},
            {'code': ['# Simple MCP server example\nfrom fastapi import FastAPI\nimport json\n\napp = FastAPI()\n\n@app.get("/tools/list")\nasync def list_tools():\n    return {\n        "tools": [\n            {"name": "get_weather", "description": "Get weather for a city"},\n            {"name": "search_docs", "description": "Search documentation"}\n        ]\n    }\n\n@app.post("/tools/execute")\nasync def execute_tool(tool_name: str, args: dict):\n    if tool_name == "get_weather":\n        return {"result": f"Weather in {args.get(\\'city\\')}: Sunny, 72°F"}\n    return {"error": "Tool not found"}\n\nprint("MCP server structure defined!")']},
            {'markdown': ['## 🔧 Building MCP Servers\n\n**Example Use Cases:**\n- Database access\n- API integrations\n- File system operations\n- Custom business logic\n']},
            {'code': ['# MCP Tool Definition\nimport json\n\nclass MCPTool:\n    def __init__(self, name, description, parameters):\n        self.name = name\n        self.description = description\n        self.parameters = parameters\n    \n    def to_json(self):\n        return {\n            "name": self.name,\n            "description": self.description,\n            "parameters": self.parameters\n        }\n\n# Define a tool\nweather_tool = MCPTool(\n    name="get_weather",\n    description="Get current weather",\n    parameters={"city": "string"}\n)\n\nprint(json.dumps(weather_tool.to_json(), indent=2))']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ MCP standardizes AI-data connections\n✅ Servers expose tools and resources\n✅ Enables modular AI systems\n✅ Foundation for enterprise AI\n']}
        ]
    },
    '06-generative-ai/09-advanced-genai-techniques.ipynb': {
        'title': '🌟 GenAI: Advanced Techniques',
        'sections': [
            {'markdown': ['## 🎯 Fine-tuning LLMs\n\n**Techniques:**\n- LoRA (Low-Rank Adaptation)\n- QLoRA (Quantized LoRA)\n- RLHF (Reinforcement Learning from Human Feedback)\n- Instruction tuning\n']},
            {'code': ['# pip install peft transformers\nfrom peft import LoraConfig, get_peft_model\n\n# LoRA configuration\nlora_config = LoraConfig(\n    r=8,  # rank\n    lora_alpha=32,\n    target_modules=["q_proj", "v_proj"],\n    lora_dropout=0.1\n)\n\nprint("LoRA config:", lora_config)']},
            {'markdown': ['## 🎓 Key Takeaways\n✅ LoRA is parameter-efficient\n✅ RLHF aligns models with human preferences\n✅ Quantization reduces model size\n']}
        ]
    },
    '06-generative-ai/10-practical-applications.ipynb': {
        'title': '🌟 GenAI: Practical Applications',
        'sections': [
            {'markdown': ['## 🚀 Building AI Applications\n\n**Complete Projects:**\n1. Document Q&A with RAG\n2. AI Agent Automation\n3. Custom Chatbot\n4. Code Assistant\n']},
            {'code': ['# pip install streamlit openai langchain\nimport streamlit as st\n\n# Simple chatbot UI\n# st.title("AI Chatbot")\n# user_input = st.text_input("You:")\n# if user_input:\n#     response = "AI response here"\n#     st.write(f"Bot: {response}")\n\nprint("Ready to build AI apps!")']},
            {'markdown': ['## 🎓 Final Takeaways\n✅ Combine RAG + Agents + MCP for powerful systems\n✅ Start simple, iterate\n✅ Focus on user needs\n✅ The AI revolution is here - build amazing things!\n\n## 🎉 Congratulations!\nYou completed the entire Python course from basics to cutting-edge GenAI!\n']}
        ]
    },
}

for filepath, config in notebooks.items():
    nb = create_notebook(config['title'], config['sections'])
    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'✅ Created {filepath}')

print('\n🎉 ALL NOTEBOOKS CREATED!')
print('🌟 Course is complete: 37 notebooks covering Basics → GenAI with RAG, Agents, MCP!')
