import time

import streamlit as st

# Set page configuration
st.set_page_config(page_title="Flux.1 Personalization Service", layout="wide")

# Custom CSS to improve the look of the app
st.markdown(
    """
    <style>
    .stButton>button {
        width: 100%;
    }
    .stProgress>div>div>div {
        background-color: #1E90FF;
    }
    .image-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .image-container img {
        width: 32%;
        border-radius: 10px;
    }
    .image-caption {
        text-align: center;
        font-style: italic;
        font-size: 0.8em;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Dummy data
paris_prompts = [
    "A futuristic Eiffel Tower in a cyberpunk Paris",
    "Parisian café on Mars with alien croissants",
    "Louvre pyramid as a holographic art gallery",
    "Versailles gardens with bioluminescent plants",
    "Flying cars racing down Champs-Élysées",
]

training_modes = ["Cyberpunk", "Cosmic", "Biopunk", "Steampunk", "Solarpunk"]

# Initialize session state
if "trained_models" not in st.session_state:
    st.session_state.trained_models = []
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Dummy user database
users = {"demo": "password", "user": "password"}


# Function to simulate image generation
def generate_image(prompt):
    time.sleep(2)  # Simulate processing time
    image = "https://i.postimg.cc/PqgR9mc1/56fd4a9a-2bce-422b-a90b-52e33bd92cf3.jpg"
    return image


# Function to display inspiration images
def display_inspiration_images():
    st.markdown("### Inspiration")
    images = [
        ("https://i.redd.it/yf8ws9mv8e621.jpg", "Cyberpunk Paris"),
        (
            "https://assets.bonappetit.com/photos/605218873b0236be8081d87e/16:9/w_1920,c_limit/Mars_2112_interior-banqutte_daroff-design.jpg",
            "Martian Café",
        ),
        (
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFA2EBw_Sj9qJrCaub1I29UtTQ8WFvLfiqgA&s",
            "Futuristic Versailles",
        ),
    ]

    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Display each image in its respective column
    with col1:
        st.image(images[0][0], caption=images[0][1], use_column_width=True)
    with col2:
        st.image(images[1][0], caption=images[1][1], use_column_width=True)
    with col3:
        st.image(images[2][0], caption=images[2][1], use_column_width=True)

    # Add some spacing
    st.write("")


# Authentication functions
def login():
    st.subheader("Login")
    username = st.text_input("Username", value="demo")
    password = st.text_input("Password", value="password", type="password")
    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Logged in as {username}")
        else:
            st.error("Invalid username or password")


def signup():
    st.subheader("Sign Up")
    new_username = st.text_input("Choose a username")
    new_password = st.text_input("Choose a password", type="password")
    if st.button("Sign Up"):
        if new_username and new_password:
            users[new_username] = new_password
            st.success("Account created successfully! Please log in.")
        else:
            st.error("Please provide both username and password")


# Training mode
def training_mode():
    st.header("Training Thousands of Personalized Flux.1 Models")
    display_inspiration_images()

    # Step 1: Upload images
    st.subheader("Step 1: Upload Training Images")
    uploaded_files = st.file_uploader(
        "Choose images for training", accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} images uploaded successfully!")

        # Step 2: Train model
        st.subheader("Step 2: Train Your Personalized Model")
        model_name = st.text_input("Enter a name for your model")
        training_mode = st.selectbox("Select a training mode", training_modes)

        cloud_provider = st.selectbox(
            "Select cloud provider for training", ["AWS", "GCP", "Azure"]
        )

        if st.button("Start Training") and model_name:
            with st.spinner(
                f"Training your personalized {training_mode} model..."
            ):
                # Simulate training process
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.05)
                    progress_bar.progress(i + 1)
            st.success(
                f"Training completed successfully! Model '{model_name}' ({training_mode}) is now available."
            )
            st.session_state.trained_models.append(
                f"{model_name} ({training_mode})"
            )

            # Display additional information
            st.info(f"Model trained on {cloud_provider}")
            st.info("Compliance check: Model adheres to EU AI Act regulations")

            if st.button("Go to Inference"):
                st.session_state.mode = "Inference"
                st.experimental_rerun()


# Inference mode
def inference_mode():
    st.header("Generate Images with Your Personalized Flux.1 Model")
    display_inspiration_images()

    if not st.session_state.trained_models:
        st.warning("No trained models available. Please train a model first.")
        return

    selected_model = st.selectbox(
        "Choose a trained model", st.session_state.trained_models
    )
    selected_prompt = st.selectbox("Choose a prompt", paris_prompts)
    custom_prompt = st.text_input("Or enter your own prompt")

    final_prompt = custom_prompt if custom_prompt else selected_prompt

    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            generated_image = generate_image(final_prompt)
        st.image(
            generated_image,
            caption=f"Generated Image: {final_prompt}",
            use_column_width=True,
        )
        st.info("Image generated using the selected personalized Flux.1 model")
        st.info("Prompt and generation parameters logged for reproducibility")


# Main app
def main():
    st.title("Flux.1 Personalization Service")

    # Add information about managing thousands of model finetunes
    st.markdown("""
    ### Empowering Enterprises with Personalized AI
    Learn how to efficiently manage and deploy thousands of customized AI models for your organization:
    - Scale your AI infrastructure with cloud-agnostic solutions
    - Ensure regulatory compliance across all your models
    - Implement robust tracking for data and model lineage
    - Maintain version control for consistent and reproducible results
    - Optimize resource allocation for cost-effective model training and inference
    """)

    if not st.session_state.logged_in:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            login()
        with tab2:
            signup()
    else:
        st.sidebar.success(f"Logged in as {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.experimental_rerun()

        mode = st.sidebar.radio("Select Mode", ["Training", "Inference"])

        if mode == "Training":
            training_mode()
        else:
            inference_mode()


if __name__ == "__main__":
    main()
