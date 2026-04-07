from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
import json
from langchain_core.prompts import PromptTemplate

load_dotenv()
# create a streamlit app to test the model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-Next",    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

# load the prompt template from json file
with open("template.json") as f:
    data = json.load(f)

# create a prompt template from the loaded data
template = PromptTemplate(
    input_variables=data["input_variables"],
    template=data["template"]
)
#''' ------------------------- Streamlit App ---------------------------- '''
st.header("Research Tab")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


# without chain, we have invoke both prompt and model separately, which is not ideal. We will create a chain to connect them later.
# """prompt = template.invoke(
#         {'paper_input':paper_input,
#         'style_input':style_input,
#         'length_input':length_input})
# result = model.invoke(prompt)"""


# with chain, we can directly invoke the chain with the input variables, and it will automatically pass the output of the prompt to the model.
if st.button("Explain Paper"):
    try:
        # create a chain to connect the prompt and the model
        chain = template | model
        with st.spinner("Generating explanation..."):
         result = chain.invoke({
            'paper_input':paper_input,
            'style_input':style_input,
            'length_input':length_input    
            })
        st.subheader("research result:")
        st.write(result.content)
    except Exception as e:
        st.error(f"An error occurred: {e}")