import streamlit as st
import requests

db_url = "http://10.0.0.29:8503"

st.title("Upload Report")
st.write("This is the Upload Report page.")
 

# Dropdown menu
uploaded_file =  st.file_uploader("Choose a file")

def process_pdf(file):
    try:
        files = {"file": (file.name, file, "application/pdf")}
        response = requests.post(f"{db_url}/process-pdf", files=files)
        return response.json()
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None
    
if uploaded_file is not None:  
    with st.spinner("Processing PDF..."):
        result = process_pdf(uploaded_file)
        if result:
            message = result.get("message", "No message provided")
            st.success(f"PDF processed successfully! {message}")
        else:
            st.error("Failed to process PDF.")
            
# Add some usage instructions
with st.sidebar:
    st.markdown("""
    ### How to use
    1. Enter patient name or ID in the search box
    2. Click on patient name to view preview
    3. Click "View Details" to see complete patient information
    """)
 