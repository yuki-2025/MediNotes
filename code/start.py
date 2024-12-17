import streamlit as st

if "logged_in" not in st.session_state:
    st.session_state.logged_in = True

# def login():
#     if st.button("Log in"):
#         st.session_state.logged_in = True
#         st.rerun()

def logout():
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()

# login_page = st.Page(login, title="Log in", icon=":material/login:")
 
def initialize_pages():
    
    logout_page = st.Page(logout, title="Log out", icon=":material/logout:")
    search = st.Page("search.py", title="1. Search Patient", icon=":material/search:")
    patient_info = st.Page("patient_info.py", title="2. Patient Information", icon=":material/person:")
    consultation = st.Page("app.py", title="3. Consultation", icon=":material/medical_services:")
    upload_report = st.Page("upload_report.py", title="4. Upload Report", icon=":material/upload_file:") 
    return logout_page, search, patient_info, consultation, upload_report 

if st.session_state.logged_in:
    logout_page, search, patient_info, consultation, upload_report = initialize_pages()  # Call initialization function
    pg = st.navigation(
        {
            "Account": [logout_page],
            "Doctor Workflow": [search, patient_info, consultation, upload_report], 
        }
    )
# else:
#     pg = st.navigation([login_page])

pg.run()
