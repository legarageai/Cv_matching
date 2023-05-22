import streamlit as st
from pathlib import Path


st.title("Demo")
# st.image(res, width = 800)

st.markdown("**S'il vous plaît remplissez lz formulaire suivant :**")
with st.form(key="Formulaire :", clear_on_submit = True):
    Name = st.text_input("Nom: ")
    Email = st.text_input("Votre Email : ")
    File = st.file_uploader(label = "Upload file", type=["pdf","docx"])
    Submit = st.form_submit_button(label='Submit')
    

st.subheader("Details : ")
st.metric(label = "Nom:", value = Name)
st.metric(label = "Email :", value = Email)

if Submit :
    st.markdown("**The file is sucessfully Uploaded.**")

    # Save uploaded file to 'F:/tmp' folder.
    save_folder = 'Cv_matching-main\CV matching\data\CV'
    save_path = Path(save_folder, File.name)
    with open(save_path, mode='wb') as w:
        w.write(File.getvalue())

    if save_path.exists():
        st.success(f'File {File.name} est sauvegardé!')