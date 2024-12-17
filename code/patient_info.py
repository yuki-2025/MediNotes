import streamlit as st
import pandas as pd
import plotly.graph_objects as go 
import base64
import os
from PIL import Image
import io


def get_pdf_download_link(file_path, link_text):
    with open(file_path, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="soap_note.pdf">{link_text}</a>'
    return href


def get_pdf_display(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    return pdf_display



def resize_image(image_path):
    STANDARD_SIZE = (300, 300)  # You can adjust this size as needed
    with Image.open(image_path) as img:
        img = img.resize(STANDARD_SIZE, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    return buf

def display_patient_info(): 
    if "selected_patient" not in st.session_state or "selected_patient_id" not in st.session_state:
        st.error("No patient selected. Please select a patient from the search page.")
        return

    patient_data = st.session_state.selected_patient
    patient_id = st.session_state.selected_patient_id
    
    st.title(f"Patient Information: {patient_data['name']}")
    
    if 'long_summary' in patient_data:
        st.subheader("Summary")
        st.info(patient_data['long_summary'])

    # Create three columns for basic info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("Personal Information")
        st.write(f"**ID:** {patient_id}")
        st.write(f"**Name:** {patient_data['name']}")
        st.write(f"**Age:** {patient_data['age']}")
        st.write(f"**Gender:** {patient_data['gender']}")
        st.write(f"**Blood Type:** {patient_data['blood_type']}")

    with col2:
        st.subheader("Medical Conditions")
        for condition in patient_data['conditions']:
            st.write(f"• {condition}")

    with col3:
        st.subheader("Current Medications")
        for medication in patient_data['medications']:
            st.write(f"• {medication}")
    
    with col4:
        st.subheader("History of Past Illness")
        for illness in patient_data['history']:
            st.write(f"• {illness}")

    st.divider()
    
    # Visit History
    st.subheader("Visit History")
    if 'visit_history' in patient_data:
        visits_data = patient_data['visit_history']
        
        # Create columns for the table
        cols = st.columns([2, 3, 1])
        
        # Header row
        cols[0].write("**Visit Date**")
        cols[1].write("**Chief Complaint**")
        cols[2].write("**SOAP Note**")
        
        # Data rows
        for visit in visits_data:
            col1, col2, col3 = st.columns([2, 3, 1])
            col1.write(visit['date'])
            col2.write(visit['chief_complaint'])
            if col3.button("View", key=f"soap_{visit['date']}"):
                st.session_state.selected_visit = visit

        # Display selected SOAP note
        if 'selected_visit' in st.session_state:
            with st.expander("SOAP Note Details", expanded=True):
                st.write(f"Displaying SOAP Note for visit on {st.session_state.selected_visit['date']}")
                pdf_display = get_pdf_display(f"data/{st.session_state.selected_visit['note']}")
                # pdf_display = get_pdf_display("app/generated_document.pdf")
                st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.write(f"Last Visit: {patient_data['last_visit']}")

    # Health Metrics Charts
    if 'blood_pressure' in patient_data and 'blood_sugar' in patient_data:
        st.subheader("Health Metrics")
        
        tab1, tab2 = st.tabs(["Blood Pressure", "Blood Sugar"])
        
        with tab1:
            fig_bp = go.Figure()
            fig_bp.add_trace(go.Scatter(y=patient_data['blood_pressure'], 
                                      mode='lines+markers',
                                      name='Blood Pressure'))
            fig_bp.update_layout(title='Blood Pressure History',
                               yaxis_title='Blood Pressure (mmHg)',
                               xaxis_title='Measurement Number')
            st.plotly_chart(fig_bp, use_container_width=True)

        with tab2:
            fig_bs = go.Figure()
            fig_bs.add_trace(go.Scatter(y=patient_data['blood_sugar'], 
                                      mode='lines+markers',
                                      name='Blood Sugar'))
            fig_bs.update_layout(title='Blood Sugar History',
                               yaxis_title='Blood Sugar (mg/dL)',
                               xaxis_title='Measurement Number')
            st.plotly_chart(fig_bs, use_container_width=True)

    st.divider()
     
    if 'images' in patient_data:
        st.subheader("Medical Images")
        STANDARD_SIZE = (300, 300)  # You can adjust this size as needed

        # Calculate the number of rows needed
        num_images = len(patient_data['images'])
        num_rows = (num_images + 2) // 3  # Round up to the nearest integer

        for row in range(num_rows):
            cols = st.columns(3)
            for col in range(3):
                idx = row * 3 + col
                if idx < num_images:
                    image_data = patient_data['images'][idx]
                    with cols[col]:
                        image_path = os.path.join(os.path.dirname(__file__),  'data', image_data['image'])
                        if os.path.exists(image_path):
                            # image = Image.open(image_path)
                            resized_image = resize_image(image_path)
                            st.image(resized_image.getvalue(), caption=image_data['type'], use_container_width =True)
                            with st.expander(f"{image_data['date']} - {image_data['type']}"):
                                st.write(f"**Date:** {image_data['date']}")
                                st.write(f"**Type:** {image_data['type']}")
                                st.write(f"**Finding:** {image_data['finding']}")
                        else:
                            st.error(f"Image file not found: {image_data['image']}")
                            
    if st.button("Continue Consultation", key=f"view_{patient_id}"):
        st.switch_page("app.py")
                    
with st.sidebar:
    st.markdown("""
    ### How to use
    1. Check the patient information.
    2. Click "Continue Consultation" to go to the next page
    """)
 
display_patient_info()