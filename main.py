import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Agente Inteligente con Groq",
    page_icon="🤖",
    layout="centered"
)

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🤖 Agente Inteligente con Groq y LangChain")
st.write("Este es un agente de IA que utiliza el modelo Llama3-8b-8192 a través de la API de Groq. Puede mantener conversaciones y buscar información actualizada en la web.")

# --- INICIALIZACIÓN DEL MODELO Y HERRAMIENTAS ---
try:
    # Inicializa el modelo de Groq usando la API Key de los secrets de Streamlit
    llm = ChatGroq(
        model_name="Llama3-8b-8192",
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

    # Inicializa la herramienta de búsqueda (Tavily)
    # Asegúrate de tener TAVILY_API_KEY en tus secrets
    search_tool = TavilySearchResults(
        max_results=2,
        tavily_api_key=st.secrets["TAVILY_API_KEY"]
    )
    tools = [search_tool]

    # --- DEFINICIÓN DEL PROMPT DEL AGENTE ---
    # Este prompt le dice al agente cómo comportarse y cómo usar las herramientas.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil y amigable. Puedes usar herramientas para buscar información actual si es necesario."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # --- CREACIÓN DEL AGENTE Y EL EJECUTOR ---
    # Creamos el agente que puede decidir qué herramienta usar (tool calling)
    agent = create_tool_calling_agent(llm, tools, prompt)

    # El AgentExecutor es el que realmente ejecuta el agente y las herramientas
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True # Muestra los pensamientos del agente en la terminal
    )

    # --- GESTIÓN DEL HISTORIAL DE CHAT ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- INTERFAZ DE CHAT ---
    # Muestra los mensajes anteriores
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)

    # Input del usuario
    user_input = st.chat_input("Hazme una pregunta...")

    if user_input:
        # Muestra el mensaje del usuario
        with st.chat_message("Human"):
            st.markdown(user_input)
        
        # Añade el mensaje del usuario al historial
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        # Muestra un spinner mientras el agente piensa
        with st.spinner("El agente está pensando..."):
            # Invoca al agente con el input y el historial
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })
        
        # Muestra la respuesta del agente
        with st.chat_message("AI"):
            st.markdown(response["output"])
        
        # Añade la respuesta del agente al historial
        st.session_state.chat_history.append(AIMessage(content=response["output"]))

except KeyError as e:
    st.error(f"Error: La clave secreta '{e.args[0]}' no fue encontrada. Por favor, configúrala en los secrets de Streamlit.")
except Exception as e:
    st.error(f"Ha ocurrido un error inesperado: {e}")

