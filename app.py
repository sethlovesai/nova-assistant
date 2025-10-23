import streamlit as st
from main import get_response, get_voice_message

st.title("Nova")

if 'session_id' not in st.session_state:
    st.session_state.session_id = "user1"

query = st.text_input("Ask Timmy:")

# if st.button("Submit") and query:
#     response = get_response(query, session_id=st.session_state.session_id)
#     st.write("Question:", query)
#     st.write(response)

if st.button('Submit') and query: 
    response = get_response(query, session_id=st.session_state.session_id)
    placeholder = st.empty()
    streamed_text = ""
    for word in response:
        streamed_text += word
        placeholder.markdown(streamed_text)
    
    get_voice_message(streamed_text)

    