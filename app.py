import gradio as gr
import torch
import KG
from huggingface_hub import InferenceClient
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer

####################################### KNOWLEDGE GRAPH ########################################
graph,df = KG.build_graph_from_csv(path="combined.csv")

model_name = "all-MiniLM-L6-v2" 
embedder = SentenceTransformer(model_name)

texts = KG.triples_to_texts(df)
embs = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

index,index_to_triple = KG.build_faiss_index(embs,texts)

def graphRAG(query,G=graph,d=df):
    hits, idxs = KG.retrieve_top_k(query,embedder=embedder,index=index,index_to_triple=index_to_triple,k=10)
    SG, hit_triples = KG.build_subgraph_from_hits(G, d, hits)
    ctx = KG.triples_to_context(hit_triples)

    print(f"context received: {ctx}")

    return ctx
    
################################################################################################

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,               # Load in 4-bit
    bnb_4bit_use_double_quant=True,  # Use nested quantization
    bnb_4bit_quant_type="nf4",       # Use 4-bit Normal Float (nf4)
    bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computations
)

model_name = "mugemafrancisryan/mistral"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
    #device_map="auto",
    #quantization_config=bnb_config
    )
tokenizer = AutoTokenizer.from_pretrained(model_name)


pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1024)

system_message = "Hello i am your AI Recovery companion here to guide you through this journey"


def respond(
    message,
    history: list[dict[str, str]],
    system_message=system_message,
):
    ctx=graphRAG(message)

    system_prompt =  f"""
    You are a compassionate, empathetic, acction oriented addiction recovery counselor trained to support individuals in their journey to overcoming substance use or behavioral addictions.
    Your role is to provide guidance, listening, and practical strategies to help the person navigate challenges, explore underlying issues, and find healthier ways to cope.
    When responding, always create a non-judgmental, safe space where the individual feels heard, supported, and empowered.
    
    Your responses should be based on evidence-based practices for addiction recovery, such as cognitive-behavioral techniques, motivational interviewing, and mindfulness strategies. 
    You should also encourage self-compassion, self-reflection, and resilience, and help the individual set realistic, achievable goals for their recovery.
    
    Don't talk in third person and explain to the person as if they are a patient.
    Your statement to the patient should be grounded based on the context provided
    
    This is the context retrieved from the literature:{ctx}
    """

    print(system_prompt)
    
    separator_tag = "[/INST]"
    if len(history) == 0:
        history = [{"role": "system", "content": system_message}]
    else:
        history.append({"role": "user", "content": message})

    print("Prompting the system")
    result=pipe(f"<s>[INST] <<SYS>> { system_prompt } <</SYS>> { message } [/INST]")
    print(result[0]['generated_text'])
    response = result[0]['generated_text'].split(separator_tag)[1].strip()

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
chatbot = gr.ChatInterface(
    respond,
    type="messages"
)

with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.LoginButton()
    chatbot.render()


if __name__ == "__main__":
    demo.launch()
