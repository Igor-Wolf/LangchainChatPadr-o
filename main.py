from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Conectando ao modelo local
llm = Ollama(model="gemma3:12b")

# Memória
memory = ConversationBufferMemory()

# Prompt customizado
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
Você é um assistente de inteligência artificial prestativo, direto e fala em português.
Use o histórico da conversa para manter o contexto.

Histórico da conversa:
{history}

Usuário: {input}
IA:"""
)

# Loop de interação com streaming
print("🤖 Olá! Sou um assistente com resposta em tempo real. Digite 'sair' para encerrar.\n")

while True:
    user_input = input("Você: ")
    if user_input.strip().lower() == "sair":
        print("IA: Até mais! 👋")
        break

    # Gera o prompt completo com histórico
    prompt_text = prompt.format(
        history=memory.buffer,
        input=user_input
    )

    print("IA: ", end="", flush=True)

    # Streaming da resposta token por token
    response = ""
    for chunk in llm.stream(prompt_text):
        print(chunk, end="", flush=True)
        response += chunk

    print()  # Nova linha após resposta completa

    # Atualiza a memória com o turno atual
    memory.save_context({"input": user_input}, {"output": response})
