from groq import Groq
import base64
import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client with API key
client = Groq(api_key=st.secrets["apikey"])

def analyze_ingredient(image_bytes):
    # Convert the image to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # Send request to Groq API to identify ingredients
    response = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                     "type": "text",
                     "text": "Identify the ingredients in this image. 'Only the ingredients' comma separated and nothing else."
                  },
                    {
                     "type": "image_url",
                     "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                     }
                  }
                ]
            }
        ],
        stream=False,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stop=None,
    )

    return response.choices[0].message.content

def suggest_recipe(ingredients):
    # Send identified ingredients to Llama3.2 to get recipe suggestions
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "user", "content": f"Suggest a recipe using these ingredients: {ingredients} along with the approximiate calories. No need to have all ingredients. you can use few ingredients from all."}
        ]
    )
    return response.choices[0].message.content

def chat_with_ai(prompt, ingredients):
    # Send user prompt to Llama3.2 to get a response
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": f"{prompt} (Ingredients: {ingredients})"}
        ]
    )
    return response.choices[0].message.content

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize flag to show chat option
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False

# Initialize initial response
if 'initial_response' not in st.session_state:
    st.session_state.initial_response = None

# Streamlit UI
st.title("AI Cooking Assistant")
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .chat-bubble {
        background-color: #DCF8C6;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-bubble {
        background-color: #E0F7FA;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload ingredient images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_files:
    ingredients_list = []

    for uploaded_file in uploaded_files:
        # Analyze each uploaded image
        image_bytes = uploaded_file.read()
        ingredients = analyze_ingredient(image_bytes)
        ingredients_list.append(ingredients)
        st.write(f"Identified Ingredients in {uploaded_file.name}: {ingredients}")

    # Suggest recipes based on the identified ingredients
    if ingredients_list:
        all_ingredients = ", ".join(ingredients_list)
        st.write(f"All identified ingredients: {all_ingredients}")
        
        if st.session_state.initial_response is None:
            st.session_state.initial_response = suggest_recipe(all_ingredients)
        
        st.write("Suggested Recipe:")
        st.write(st.session_state.initial_response)
        
        # Set flag to show chat option
        st.session_state.show_chat = True

# Display chat history with WhatsApp-like styling
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"<div class='user-bubble'>User: {message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble'>AI: {message['content']}</div>", unsafe_allow_html=True)

# Chat option for user interaction
if st.session_state.show_chat:
    st.write("Chat with AI about the recipe:")
    user_input = st.text_input("Ask a question about the recipe:")

    if st.button("Get Response"):
        if user_input:
            # Add user input to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get AI response
            chat_response = chat_with_ai(user_input, all_ingredients)
            
            # Add AI response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": chat_response})
            
            # Display chat history
            #st.experimental_rerun()
