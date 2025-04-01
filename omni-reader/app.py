"""This is the Streamlit UI for the OCR Extraction workflow."""

import base64
import os
import time

import streamlit as st
from PIL import Image

from run_compare_ocr import run_ocr_from_ui
from utils.model_configs import DEFAULT_MODEL, MODEL_CONFIGS


def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="OCR Extraction",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def load_model_logos():
    """Load model logos from assets folder."""
    logo_mapping = {}
    logos_dir = "./assets/logos"

    # Try to load specific model logos first, fall back to provider logos
    for _, model_config in MODEL_CONFIGS.items():
        logo_filename = model_config.logo

        # Check if specific model logo exists
        if os.path.exists(os.path.join(logos_dir, logo_filename)):
            try:
                logo_mapping[model_config.display] = base64.b64encode(
                    open(os.path.join(logos_dir, logo_filename), "rb").read()
                ).decode()
            except Exception as e:
                print(f"Error loading logo for {model_config.display}: {e}")
                # Fall back to provider logo
                provider = model_config.provider
                if os.path.exists(os.path.join(logos_dir, f"{provider}.svg")):
                    logo_mapping[model_config.display] = base64.b64encode(
                        open(os.path.join(logos_dir, f"{provider}.svg"), "rb").read()
                    ).decode()

    return logo_mapping


def render_header(model_logos):
    """Render the page header based on selected models."""
    if st.session_state.get("comparison_mode", False):
        # Get the selected models for comparison
        selected_models = st.session_state.get("comparison_models", [DEFAULT_MODEL.name])

        # Get display names for selected models
        display_models = [
            MODEL_CONFIGS[model_id].display for model_id in selected_models[:3]
        ]  # Show up to 3 logos

        # Create header with logos
        logo_html = (
            '<div style="display: flex; justify-content: space-between; align-items: center;">'
        )

        for model_display in display_models:
            if model_display in model_logos:
                logo_html += f'<img src="data:image/svg+xml;base64,{model_logos[model_display]}" width="40" style="margin: 0 10px; vertical-align: middle;">'

        # Adjust title based on number of models
        if len(selected_models) > 1:
            logo_html += f'<h1 style="margin: 0 15px;">OCR Model Comparison ({len(selected_models)} Models)</h1>'
        else:
            model_display = MODEL_CONFIGS[selected_models[0]].display
            logo_html += f'<h1 style="margin: 0 15px;">{model_display} OCR</h1>'

        logo_html += "</div>"

        st.markdown(logo_html, unsafe_allow_html=True)
    else:
        # Single model mode
        selected_model_id = st.session_state.get("selected_model_id", DEFAULT_MODEL.name)
        selected_model_display = MODEL_CONFIGS[selected_model_id].display
        logo_html = ""

        if selected_model_display in model_logos:
            logo_html = f'# <img src="data:image/svg+xml;base64,{model_logos[selected_model_display]}" width="40" style="vertical-align: -8px; margin-right: 10px;"> {selected_model_display} OCR'
        else:
            logo_html = f"# {selected_model_display} OCR"

        st.markdown(logo_html, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with model selection and settings."""
    with st.sidebar:
        st.header("Settings")

        # Add multi-model selection mode
        comparison_mode = st.radio(
            "Mode",
            ["Single Model", "Compare Models"],
            horizontal=True,
        )

        if comparison_mode == "Single Model":
            # Get all model display names for single selection
            model_options = [
                model_config.display for model_id, model_config in MODEL_CONFIGS.items()
            ]

            # Set default selection to DEFAULT_MODEL
            default_index = 0
            for i, option in enumerate(model_options):
                if option == DEFAULT_MODEL.display:
                    default_index = i
                    break

            model_choice = st.selectbox(
                "Select OCR Model",
                options=model_options,
                index=default_index,
            )

            # Store single model ID
            selected_model_id = None
            for model_id, model_config in MODEL_CONFIGS.items():
                if model_config.display == model_choice:
                    selected_model_id = model_id
                    break

            st.session_state["comparison_mode"] = False
            st.session_state["selected_model_id"] = selected_model_id
        else:
            # Multiselect for model comparison
            st.subheader("Models to Compare")

            # Get all model display names with their IDs
            model_display_to_id = {
                model_config.display: model_id for model_id, model_config in MODEL_CONFIGS.items()
            }

            # Use multiselect with checkboxes for better UX
            default_models = [DEFAULT_MODEL.display]
            selected_model_displays = st.multiselect(
                "Select models to compare",
                options=list(model_display_to_id.keys()),
                default=default_models,
            )

            # Convert selected display names to model IDs
            comparison_models = [
                model_display_to_id[display] for display in selected_model_displays
            ]

            # Ensure at least 1 model is selected
            if len(comparison_models) < 1:
                st.warning("Please select at least one model.")
                comparison_models = [DEFAULT_MODEL.name]

            st.session_state["comparison_mode"] = True
            st.session_state["comparison_models"] = comparison_models

        st.header("Custom Prompt")
        use_custom_prompt = st.checkbox("Use custom prompt", value=False)
        custom_prompt = (
            st.text_area(
                "Enter your custom prompt",
                "Extract and analyze all text from the image and identify any objects present.",
                height=100,
            )
            if use_custom_prompt
            else ""
        )

        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["png", "jpg", "jpeg", "webp"],
            key=st.session_state.get("file_uploader_key", "default_uploader"),
        )
        st.markdown("---")

        # Display provider information
        display_provider_info()

    return uploaded_file, custom_prompt if use_custom_prompt else None


def display_provider_info():
    """Display information about which providers are being used."""
    providers = set()
    for model_id, model_config in MODEL_CONFIGS.items():
        providers.add(model_config.provider)

    provider_names = {"ollama": "Ollama", "openai": "OpenAI", "mistral": "MistralAI"}

    provider_text = " + ".join([provider_names.get(p, p.capitalize()) for p in providers])
    st.caption(f"Powered by {provider_text}")


def add_clear_button():
    """Add a clear button to reset the app state."""
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Clear 🗑️"):
            # Create a list of all possible result keys based on model IDs
            keys_to_clear = ["ocr_result"]
            for model_id in MODEL_CONFIGS:
                result_key = f"{model_id}_result"
                keys_to_clear.append(result_key)

            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

            st.session_state["uploaded_file"] = None

            # This forces the file uploader to reset on next rerun
            st.session_state["file_uploader_key"] = str(time.time())  # Changes the key each time

            st.rerun()


def process_uploaded_image(image):
    """Process and display the uploaded image."""
    # Get the original dimensions
    img_width, img_height = image.size

    # Set maximum dimensions
    max_width = 600
    max_height = 500

    # Calculate scaling factor to maintain aspect ratio
    width_ratio = max_width / img_width
    height_ratio = max_height / img_height

    # Use the smaller ratio to ensure image fits within both dimensions
    scale_factor = min(width_ratio, height_ratio)

    # Only resize if the image is larger than our max dimensions
    if img_width > max_width or img_height > max_height:
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        image = image.resize((new_width, new_height))

    st.image(image, caption="Uploaded Image", use_container_width=False)
    return image


def check_for_error(result):
    """Check if the OCR model has an error."""
    return (
        "error" in result
        or result["raw_text"].startswith("Error:")
        or ("success" in result and result["success"] is False)
    )


def has_no_text(result):
    """Check if the OCR model has no text."""
    return result["raw_text"] == "No text found"


def display_result(label, result, proc_time):
    """Display the results of the OCR model."""
    st.subheader(f"{label} Results")
    st.text(f"Processing time: {proc_time:.2f}s")
    st.markdown("##### Extracted Text")
    text = result["raw_text"]
    if text.startswith("Error:"):
        st.error(text)
    elif text == "No text found":
        st.warning("No text found in the image")
    else:
        st.write(text)


def run_single_model(image, model_id, custom_prompt):
    """Run OCR with a single model and display results."""
    selected_model_display = MODEL_CONFIGS[model_id].display

    with st.spinner(f"Processing image with {selected_model_display}..."):
        try:
            start = time.time()
            result = run_ocr_from_ui(
                image=image,
                model=model_id,
                custom_prompt=custom_prompt,
            )
            proc_time = time.time() - start

            # Store result in session state
            st.session_state[f"{model_id}_result"] = result

            st.subheader("Extracted Text")
            text = result["raw_text"]
            if text.startswith("Error:"):
                st.error(text)
            elif text == "No text found":
                st.warning("No text found in the image")
            else:
                st.write(text)

            st.subheader("Processing Stats")
            st.text(f"Processing time: {proc_time:.2f}s")
            st.text(f"Text length: {len(result['raw_text'])} characters")

            # Display confidence if available
            if "confidence" in result and result["confidence"] is not None:
                st.text(f"Confidence: {result['confidence']:.2%}")

            # Provider information
            model_config = MODEL_CONFIGS[model_id]
            st.text(f"Provider: {model_config.provider.capitalize()}")

            return result, proc_time

        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None, 0


def create_responsive_layout(num_models):
    """Create a responsive layout based on the number of models."""
    if num_models == 1:
        return [st.container()]  # Full width for single model
    elif num_models == 2:
        return st.columns(2)  # Two equal columns
    else:
        # For 3+ models, create a responsive layout with max 3 columns per row
        col_rows = []
        for i in range(0, num_models, 3):
            remaining = min(3, num_models - i)
            col_rows.append(st.columns(remaining))
        return col_rows


def get_column_for_model(i, num_models, cols, col_rows):
    """Get the appropriate column for a model based on the layout."""
    if num_models <= 2:
        return cols[i]
    else:
        row_idx = i // 3
        col_idx = i % 3
        return col_rows[row_idx][col_idx]


def run_multiple_models(image, model_ids, custom_prompt):
    """Run OCR with multiple models and display comparison results."""
    with st.spinner("Processing image with selected models..."):
        try:
            # Create responsive layout
            num_models = len(model_ids)
            if num_models <= 2:
                cols = st.columns(num_models)
                col_rows = None
            else:
                cols = None
                col_rows = []
                for i in range(0, num_models, 3):
                    remaining = min(3, num_models - i)
                    col_rows.append(st.columns(remaining))

            # Process each model and store results
            model_results = {}
            model_times = {}
            model_errors = {}

            for i, model_id in enumerate(model_ids):
                model_config = MODEL_CONFIGS[model_id]
                model_name = model_config.display

                # Determine which column to use based on layout
                if num_models <= 2:
                    column = cols[i]
                else:
                    row_idx = i // 3
                    col_idx = i % 3
                    column = col_rows[row_idx][col_idx]

                with column:
                    with st.spinner(f"Processing with {model_name}..."):
                        start = time.time()
                        result = run_ocr_from_ui(
                            image=image,
                            model=model_id,
                            custom_prompt=custom_prompt,
                        )
                        proc_time = time.time() - start

                        model_results[model_id] = result
                        model_times[model_id] = proc_time
                        model_errors[model_id] = check_for_error(result)

                        # Store in session state
                        st.session_state[f"{model_id}_result"] = result

                        # Display result
                        display_result(model_name, result, proc_time)

            # Only show comparison stats if there's more than one model
            if len(model_ids) > 1:
                display_comparison_stats(model_ids, model_results, model_times, model_errors)

            return model_results, model_times

        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None, None


def display_comparison_stats(model_ids, model_results, model_times, model_errors):
    """Display comparison statistics for multiple models."""
    st.markdown("### Comparison Stats")

    # Determine fastest model
    valid_times = []
    for model_id in model_ids:
        if not model_errors[model_id]:
            valid_times.append((MODEL_CONFIGS[model_id].display, model_times[model_id]))

    if valid_times:
        fastest = min(valid_times, key=lambda x: x[1])
        st.write(f"🚀 Fastest model: **{fastest[0]}** ({fastest[1]:.2f}s)")

    # Show error status for each model
    error_status = []
    for model_id in model_ids:
        if model_errors[model_id]:
            error_status.append(f"{MODEL_CONFIGS[model_id].display} failed")

    if error_status:
        st.write("⚠️ " + ", ".join(error_status))

    # Compare text lengths
    text_lengths = []
    for model_id in model_ids:
        if not model_errors[model_id] and not has_no_text(model_results[model_id]):
            text_lengths.append(
                f"{MODEL_CONFIGS[model_id].display}: {len(model_results[model_id]['raw_text'])} chars"
            )

    if text_lengths:
        st.write("📝 Text lengths: " + ", ".join(text_lengths))
    else:
        st.warning("No usable text extracted by any model")


def main():
    """Main function to run the Streamlit app."""
    # Setup page configuration
    setup_page_config()

    # Load model logos
    model_logos = load_model_logos()

    # Render sidebar and get user selections
    uploaded_file, custom_prompt = render_sidebar()

    # Render header based on selected models
    render_header(model_logos)

    # Add clear button
    add_clear_button()

    st.markdown("---")
    st.markdown(
        '<p style="margin-top: -20px;">Extract structured text from images using your chosen OCR model!</p>',
        unsafe_allow_html=True,
    )

    # Process uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        processed_image = process_uploaded_image(image)

        if st.button("Extract Text 🔍", type="primary"):
            # Check if we're in comparison mode
            comparison_mode = st.session_state.get("comparison_mode", False)

            if comparison_mode:
                # Get the list of models to compare
                comparison_models = st.session_state.get("comparison_models", [DEFAULT_MODEL.name])
                run_multiple_models(processed_image, comparison_models, custom_prompt)
            else:
                # Process with a single selected model
                selected_model_id = st.session_state.get("selected_model_id", DEFAULT_MODEL.name)
                run_single_model(processed_image, selected_model_id, custom_prompt)
    else:
        st.info("Upload an image and click 'Extract Text' to see the results here.")

    # Footer
    st.markdown("---")
    st.caption("ZenOCR - Comparing LLM OCR capabilities")


if __name__ == "__main__":
    main()
