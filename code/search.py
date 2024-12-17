import streamlit as st
import pandas as pd 

# Sample patient data - in a real application, this would come from a database
SAMPLE_PATIENTS = {
    "P001": {
        "name": "Brittany Edwards",
        "age": 32,
        "gender": "Female",
        "blood_type": "A",
        "history": ["Irritable Bowel Syndrome (IBS)"],
        "conditions": ["Irritable Bowel Syndrome with Constipation", "Nausea"],
        "medications": ["Bentyl", "Zofran", "Citrucel", "Probiotics"],
        "last_visit": "2024-07-25",
        "long_summary": "Brittany Edwards, a 32-year-old female, presented with a history of IBS and current symptoms of abdominal pain, nausea, and occasional diarrhea associated with diet choices. She has been using Bentyl as needed and has been counseled to start Zofran for nausea, daily fiber supplements like Citrucel, and probiotics for gas and bloating. She denies smoking and alcohol use. Physical exams revealed a soft and nontender abdomen, with normal bowel sounds and no rectal abnormalities. The patient understands the prescribed treatment plan and has a follow-up in two weeks.",
        "visit_history": [
             {"date": "2024-07-25", "note":"brittany_edward.pdf",  "chief_complaint": "Abdominal pain, nausea, and vomiting."},
             {"date": "2023-10-20", "note":"brittany_edward.pdf", "chief_complaint": "Abdominal pain and occasional diarrhea."},
             {"date": "2019-08-15", "note":"brittany_edward.pdf", "chief_complaint": "Initial presentation of IBS symptoms."}
        ],
        "blood_pressure": [140, 135, 142, 138],  # Sample data for chart
        "blood_sugar": [120, 115, 130, 125],  # Sample data for chart 
        "images": [
            {"date": "2024-07-25", "image":"CT.jpg", "type": "CT Scan", "finding": "No abnormalities detected"},
            {"date": "2023-10-20", "image":"x_ray.jpg", "type": "X-Ray", "finding": "Mild joint space narrowing in left knee"},
            {"date": "2023-09-05", "image":"shoulder-mri-accuracy.jpg", "type": "MRI", "finding": "Rotator cuff tear in right shoulder"}
        ]
    },
    "P002": {
        "name": "Brian Smith",
        "age": 58,
        "gender": "Male",
        "blood_type": "B+",  # Made-up for completion
        "history": ["Congestive Heart Failure", "Hypertension"],
        "conditions": ["Congestive Heart Failure", "High Blood Pressure", "Pulmonary Edema"],
        "medications": ["Lasix 80 mg", "Lisinopril 20 mg"],
        "last_visit": "2024-11-18",
        "long_summary": "Brian Smith, a 58-year-old male with a history of congestive heart failure and hypertension, presented for follow-up of his chronic conditions. He reported fatigue, lightheadedness, mild chest cramps, and occasional cough. His symptoms worsened over the last five weeks, coinciding with dietary changes due to home kitchen construction. Physical examination revealed fine crackles in the lungs, a 3/6 systolic murmur, and 1+ pitting edema. Imaging indicated mild pulmonary edema and an ejection fraction of 45% with moderate mitral regurgitation. The treatment plan includes increasing Lasix to 80 mg daily, monitoring weight, and continuing Lisinopril 20 mg daily.",
        "visit_history": [
            { "date": "2024-11-18", "note": "brian_smith.pdf", "chief_complaint": "Fatigue, lightheadedness, and dietary concerns."},
            { "date": "2024-07-15", "note": "brian_smith.pdf","chief_complaint": "Routine cardiovascular checkup."}
        ],
        "blood_pressure": [140, 135, 142, 138],  # Made-up based on typical hypertension data
        "blood_sugar": [100, 105, 110, 102],  # Hypothetical normal range
        "images": [
            {"date": "2024-11-18", "image": "x_ray_chest.jpg","type": "Chest X-ray", "finding": "Mild pulmonary edema"},
            {"date": "2024-07-15", "image": "heart_echocardiogram.jpg", "type": "Echocardiogram", "finding": "Ejection fraction of 50%, mild mitral regurgitation"},
            {"date": "2023-09-05", "image":"Myxoma_CMR.gif", "type": "MRI", "finding": "Rotator cuff tear in right shoulder"}
        ]
    },
    "P003": {
        "name": "Betty Jill",
        "age": 52,
        "gender": "Female",
        "blood_type": "O+",  # Not specified in the document
        "history": ["Uterine fibroids", "Anemia", "Gallbladder removal"],
        "conditions": ["Chest pain", "Dyspepsia", "Esophagitis"],
        "medications": [],  # No medications currently prescribed
        "last_visit": "2024-11-18",
        "long_summary": "Betty Hill, a 52-year-old female, was referred to the clinic for further evaluation of esophagitis after an emergency room visit. She reported chest pain that began spontaneously three months ago and lasted for four days, which prompted her ER visit. Tests at the ER were normal, and no medications were prescribed. The chest pain and associated difficulty swallowing resolved shortly afterward. Physical examination revealed mild tenderness to palpation in the upper abdominal quadrants but no other abnormalities. The patient elected to monitor symptoms instead of proceeding with an upper endoscopy (EGD) and will follow up if symptoms recur. The patient has a history of uterine fibroids, anemia, and gallbladder removal. No issues with bowel movements or abdominal pain were reported, though she noted a stressful job transition at the time of her symptoms.",
        "visit_history": [
            { "date": "2024-11-18","note": "betty_jill.pdf", "chief_complaint": "Esophagitis and chest pain"}
        ],
        "blood_pressure": [140, 135, 142, 138],  # Sample data for chart
        "blood_sugar": [120, 115, 130, 125],  # Sample data for chart 
        "images": [
            {"date": "2023-11-15", "image":"CT.jpg", "type": "CT Scan", "finding": "No abnormalities detected"},
            {"date": "2023-10-20", "image":"x_ray.jpg", "type": "X-Ray", "finding": "Mild joint space narrowing in left knee"},
            {"date": "2023-09-05", "image":"shoulder-mri-accuracy.jpg", "type": "MRI", "finding": "Rotator cuff tear in right shoulder"}
        ]
    },
    "P004": {
        "name": "Jane Brooks",
        "age": 48,
        "gender": "Female",
        "blood_type": "O-",  # Made-up for completion
        "history": ["Constipation", "Appendectomy at 18"],
        "conditions": ["Right Finger Contusion"],
        "medications": ["Miralax", "Motrin 600 mg"],
        "last_visit": "2024-11-18",
        "long_summary": "Jane Brooks, a 48-year-old female, presented to the clinic for evaluation of right finger pain sustained while skiing. The injury occurred when a ski strap caught her finger, bending it backward. She reports pain during movement and has been managing it with ice and ibuprofen without significant relief. Physical examination revealed tenderness over the distal phalanx with no fractures detected on X-ray. The patient was diagnosed with a contusion of the right index finger and prescribed Motrin 600 mg every six hours for one week. An aluminum foam splint was provided for support. She also reported ongoing constipation managed with Miralax, and her past surgical history includes an appendectomy at 18 years of age.",
        "visit_history": [
            { "date": "2024-11-18", "note": "jane_brooks.pdf", "chief_complaint": "Right finger pain"}
        ],
        "blood_pressure": [120, 115, 118, 122],  # Made-up normal range for consistency
        "blood_sugar": [85, 90, 88, 92],  # Hypothetical normal values
        "images": [
            { "date": "2024-11-18", "image": "hand_xray.jpg","type": "X-ray", "finding": "No fractures detected"}
        ]
    },
    "P006": {
        "name": "Logan Berry",
        "age": 58,
        "gender": "Male",
        "blood_type": "A+",  # Made-up for completion
        "history": ["Type 2 Diabetes", "Hypertension", "Osteoarthritis"],
        "conditions": ["Lumbar Strain", "Hypertension", "Type 2 Diabetes"],
        "medications": ["Meloxicam 15 mg", "Metformin 1000 mg", "Lisinopril 20 mg"],
        "last_visit": "2024-10-08",
        "long_summary": "Logan Berry, a 58-year-old male, presented for evaluation of lower back pain caused by lifting heavy boxes while assisting his daughter. He reported hearing a pop in his back followed by stiffness and discomfort, particularly during movement. Physical examination revealed pain to palpation of the lumbar spine, pain with flexion and extension, and a negative straight leg raise. Imaging ruled out fractures, and the condition was diagnosed as a lumbar strain. His Type 2 diabetes and hypertension were also reviewed; hemoglobin A1c was elevated at 8.0, and blood pressure readings were slightly high. Treatment includes Meloxicam for back pain, increased Metformin dosage for diabetes, and increased Lisinopril dosage for hypertension. He was referred to physical therapy and advised to avoid strenuous activities.",
        "visit_history": [
            {  "date": "2024-11-18", "note": "logan_berry.pdf", "chief_complaint": "Back pain"}
        ],
        "blood_pressure": [140, 138, 142, 136],  # Based on hypertension trends
        "blood_sugar": [150, 145, 152, 148],  # Hypothetical readings reflecting elevated A1c
        "images": [
            {"date": "2024-07-25", "image":"CT.jpg", "type": "CT Scan", "finding": "No abnormalities detected"},
            {"date": "2023-10-20", "image":"x_ray.jpg", "type": "X-Ray", "finding": "Mild joint space narrowing in left knee"},
            {"date": "2023-09-05", "image":"shoulder-mri-accuracy.jpg", "type": "MRI", "finding": "Rotator cuff tear in right shoulder"}
        ]
    }
}

st.title("Patient Search")

# Search input
search_query = st.text_input("Search for patient by name or ID")

if search_query:
    # Search logic
    results = []
    search_query = search_query.lower()
    
    for patient_id, data in SAMPLE_PATIENTS.items():
        if (search_query in patient_id.lower() or 
            search_query in data['name'].lower()):
            results.append((patient_id, data))
    
    if results:
        st.subheader("Search Results")
        for patient_id, patient_data in results:
            with st.expander(f"{patient_data['name']} ({patient_id})"):
                # Display patient preview in columns
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.write("**Personal Information**")
                    st.write(f"Age: {patient_data['age']}")
                    st.write(f"Gender: {patient_data['gender']}")
                    st.write(f"Blood Type: {patient_data['blood_type']}")
                
                with col2:
                    st.write("**Medical Information**")
                    st.write("Conditions:")
                    for condition in patient_data['conditions']:
                        st.write(f"- {condition}")
                
                with col3:
                    if st.button("View Details", key=f"view_{patient_id}"):
                        st.session_state.selected_patient = patient_data
                        st.session_state.selected_patient_id = patient_id
                        st.switch_page("patient_info.py")

    else:
        st.warning("No patients found matching your search criteria.")


    
    
# Add some usage instructions
with st.sidebar:
    st.markdown("""
    ### How to use
    1. Enter patient name or ID in the search box
    2. Click on patient name to view preview
    3. Click "View Details" to see complete patient information
    """)
 