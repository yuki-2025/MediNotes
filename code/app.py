import streamlit as st
from langchain.llms import OpenAI
from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

st.title('🦜🔗 Medinotes')

items = ["Yuki20/llama3_8b_aci", "Yuki20/alpaca7B-lora", "LLaMA3"]

# Title of the sidebar
st.sidebar.title("Model Selection")

# Dropdown menu
selected_item = st.sidebar.selectbox("Select an option", items)
 
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Summarize medical dialogues using the SOAP note format, which categorizes medical information into distinct sections, each with specific subdivisions to ensure a comprehensive evaluation and treatment plan. The "Subjective" section comprises the "Chief Complaint (CC)," succinctly noting the primary reason for the visit, such as "knee pain." It also includes the "History of Present Illness (HPI)," detailing the specifics of the illness including its onset, duration, and characteristics, and the "Past Medical History," which records significant historical medical events like surgeries or chronic conditions.

The "Objective" section is divided into "Physical Exam (PE)" and "Findings." The "Physical Exam" details observations and measurements taken during the physical examination, focusing on areas such as the knee and noting any abnormalities or symptoms. The "Findings" segment includes results from diagnostic procedures, such as X-rays, specifically noting critical findings like the absence of fractures or dislocations.

Lastly, the "Assessment and Plan" section encapsulates the diagnosis in the "Assessment" subsection, based on the synthesis of subjective complaints and objective findings. The "Plan" subsection outlines the proposed management strategy, including further diagnostic tests, treatment options, and follow-up measures necessary to address the patient’s condition. This structured approach ensures a detailed and organized method for documenting patient care and planning appropriate interventions.

### Input:
{}

### Response:
{}"""

@st.cache_resource
def load_model_and_tokenizer(model_name):

    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(selected_item)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

def generate_response(input_text):
    inputs = tokenizer(
    [
        alpaca_prompt.format(
           input_text, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    text_streamer = TextStreamer(tokenizer)
    result = model.generate(**inputs, streamer=text_streamer, max_new_tokens=500, use_cache=True)
    return tokenizer.decode(result[0], skip_special_tokens=True)


with st.form('my_form'):

    text = st.text_area('Enter text:', "[doctor] hi , bryan . how are you ? [patient] i'm doing well . i'm a little sore . [doctor] yeah ? [patient] yeah . [doctor] all right , well , i know the nurse told you about dax . i'd like to tell dax a little bit about you , okay ? [patient] that's fine . [doctor] so bryan is a 55-year-old male with a past medical history significant for prior discectomy , who presents with back pain . so , bryan , what happened to your back ? [patient] you ... my wife made me push a , uh , refrigerator out through the other room , and when i was helping to move it , i felt something in my back on the lower right side . [doctor] okay , on the lower right side of this back ? [patient] yes . [doctor] okay . those wives , always making you do stuff . [patient] yes . [doctor] and what day did this happen on ? how long ago ? [patient] uh , this was about five days ago . [doctor] five days ago . [patient] and , you know , i have that history of discectomy . [doctor] yeah . [patient] and i'm just worried that something happened . [doctor] okay . all right . and , and what have you taken for the pain ? [patient] um , i have , uh , been taking ibuprofen . uh , and i tried once tylenol and ibuprofen at the same time , and that gave me some relief . [doctor] okay . all right . and have you had any symptoms like pain in your legs or numbing or tingling ? [patient] um , no , nothing significant like that . [doctor] okay , just the pain in your back . [patient] just the pain in the back . it hurts to bend over . [doctor] okay , and any problems with your bladder or your bowels ? [patient] no , no . [doctor] i know the nurse said to review a symptom sheet when you checked in . [patient] mm-hmm . [doctor] and i know that you were endorsing the back pain . any other symptoms ? chest pain ? shortness of breath ? abdominal pain ? [patient] no . [doctor] nausea ? vomiting ? [patient] no other symptoms . [doctor] okay . all right . well , let's go ahead and do a quick physical exam . hey , dragon , show me the vital signs . so your vital signs here in the office look really good . you do n't have a fever . your blood pressure's nice and controlled . so that ... that's good . i'm just gon na check out your back and your heart and your lungs , okay ? [patient] okay . [doctor] okay , so on physical examination , you know , your heart sounds great . there's ... it's a regular rate and rhythm . your lungs are nice and clear . on your back exam , you do have some pain to palpation of the right lumbar spine , uh , in the paraspinal muscles along with decreased flexion and extension of the back , and you have a positive straight leg on the right . or positive straight leg raise on the right , uh , but your strength is good bilaterally in your lower extremities . so that means that i think that you've injured your back . [patient] okay . [doctor] uh , but , you know , i think it's something that we can , we can fix , okay ? [patient] okay , you do n't think there's anything wrong with the ... where i had the surgery before . [doctor] i do n't think so . [patient] okay . [doctor] let's took at some of your results . hey , dragon , show me the back x-ray . so this is an x-ray of your lumbar spine . you know , there's good bony , bony alignment . i do n't see any fracture or anything like that . so that's a good sign . um , hey , dragon . show me the labs . and your labs here all look good , so i'm , i'm happy to see that . uh , so let's talk a little bit about my assessment and my plan for you , okay ? [patient] okay . [doctor] so i ... my assessment for your first problem , your back pain . i think you have a lumbar strain . i do n't think that anything else is going on , but i wan na go ahead and order an mri- [patient] okay . [doctor] just to be sure . [patient] okay . [doctor] okay ? and then i'm gon na prescribe you some meloxicam 15 milligrams once a day along with some ultram , 50 milligrams every four hours as needed , okay ? [patient] okay . [doctor] um , and then we'll go ahead and refer you to some physical therapy once we get the mri results back , okay ? [patient] should i continue to take the tylenol and the ibuprofen ? [doctor] you can stop the ibuprofen . [patient] okay . [doctor] you can take tylenol if you want . [patient] okay . [doctor] you know to call me if , if you need anything . [patient] okay . [doctor] okay ? [patient] okay . [doctor] any questions , uh , bryan ? [patient] no , no questions . [doctor] okay . hey , dragon , finalize the note .")
    submitted = st.form_submit_button('Submit') 
    if submitted:
        response = generate_response(text)
        st.info(response)
