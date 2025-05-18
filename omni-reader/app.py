"""This is the Streamlit UI for the OCR Extraction workflow."""

import base64
import os
import time

import streamlit as st
from PIL import Image

from utils.model_configs import DEFAULT_MODEL, MODEL_CONFIGS
from utils.ocr_processing import run_ocr


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
    processed_displays = set()

    for _, model_config in MODEL_CONFIGS.items():
        if model_config.display in processed_displays:
            continue

        processed_displays.add(model_config.display)
        logo_filename = model_config.logo

        if os.path.exists(os.path.join(logos_dir, logo_filename)):
            try:
                logo_mapping[model_config.display] = base64.b64encode(
                    open(os.path.join(logos_dir, logo_filename), "rb").read()
                ).decode()
            except Exception as e:
                print(f"Error loading logo for {model_config.display}: {e}")
                provider = model_config.shorthand
                if os.path.exists(os.path.join(logos_dir, f"{provider}.svg")):
                    logo_mapping[model_config.display] = base64.b64encode(
                        open(
                            os.path.join(logos_dir, f"{provider}.svg"), "rb"
                        ).read()
                    ).decode()

    return logo_mapping


def render_header(model_logos=None):
    """Render the page header based on selected models."""
    if st.session_state.get("comparison_mode", False):
        selected_models = st.session_state.get(
            "comparison_models", [DEFAULT_MODEL.name]
        )

        if len(selected_models) > 1:
            st.title(f"OCR Model Comparison ({len(selected_models)} Models)")
        else:
            model_display = MODEL_CONFIGS[selected_models[0]].display
            st.title(f"{model_display} OCR")
    else:
        selected_model_id = st.session_state.get(
            "selected_model_id", DEFAULT_MODEL.name
        )
        selected_model_display = MODEL_CONFIGS[selected_model_id].display

        if model_logos and selected_model_display in model_logos:
            logo_html = f'# <img src="data:image/svg+xml;base64,{model_logos[selected_model_display]}" width="40" style="vertical-align: -8px; margin-right: 10px;"> {selected_model_display} OCR'
            st.markdown(logo_html, unsafe_allow_html=True)
        else:
            st.title(f"{selected_model_display} OCR")


def render_sidebar():
    """Render the sidebar with model selection and settings."""
    with st.sidebar:
        st.header("Settings")

        comparison_mode = st.radio(
            "Mode",
            ["Single Model", "Compare Models"],
            horizontal=True,
        )

        # Get unique models (avoid duplicates from shorthand keys)
        unique_models = {
            model_id: model_config
            for model_id, model_config in MODEL_CONFIGS.items()
            if model_id == model_config.name
        }

        if comparison_mode == "Single Model":
            model_options = [
                model_config.display
                for model_id, model_config in unique_models.items()
            ]

            # Find default model index
            default_index = 0
            for i, option in enumerate(model_options):
                if option == DEFAULT_MODEL.display:
                    default_index = i
                    break

            model_choice = st.selectbox(
                "Select OCR Model",
                options=model_options,
                index=default_index,
                key="model_selector_single",
            )

            # Map display name back to model ID
            selected_model_id = None
            for model_id, model_config in unique_models.items():
                if model_config.display == model_choice:
                    selected_model_id = model_id
                    break

            st.session_state["comparison_mode"] = False
            st.session_state["selected_model_id"] = selected_model_id
        else:
            # Multiselect for model comparison
            st.subheader("Models to Compare")
            model_display_to_id = {
                model_config.display: model_id
                for model_id, model_config in unique_models.items()
            }
            default_models = [DEFAULT_MODEL.display]
            previously_selected = st.session_state.get(
                "selected_model_displays", default_models
            )
            multiselect_key = f"model_multiselect_{len(previously_selected)}"

            selected_model_displays = st.multiselect(
                "Select models to compare",
                options=list(model_display_to_id.keys()),
                default=previously_selected,
                key=multiselect_key,
            )

            if not selected_model_displays:
                st.warning(
                    "At least one model must be selected. Using default model."
                )
                selected_model_displays = default_models

            if selected_model_displays != previously_selected:
                st.session_state["selected_model_displays"] = (
                    selected_model_displays
                )
                st.rerun()

            comparison_models = [
                model_display_to_id[display]
                for display in selected_model_displays
            ]

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
            type=["png", "jpg", "jpeg", "webp", "tiff"],
            key=st.session_state.get("file_uploader_key", "default_uploader"),
        )

        # Store uploaded file in session state
        if uploaded_file is not None:
            st.session_state["uploaded_file"] = uploaded_file

        st.markdown("---")
        display_provider_info()

    return st.session_state.get(
        "uploaded_file", uploaded_file
    ), custom_prompt if use_custom_prompt else None


def display_provider_info():
    """Display information about which providers are being used."""
    providers = set()
    for model_id, model_config in MODEL_CONFIGS.items():
        if model_id == model_config.name:  # Skip shorthand duplicates
            providers.add(model_config.ocr_processor)

    provider_names = {
        "ollama": "Ollama",
        "openai": "OpenAI",
        "mistral": "MistralAI",
    }
    provider_text = " + ".join(
        [provider_names.get(p, p.capitalize()) for p in providers]
    )
    st.caption(f"Powered by {provider_text}")


def add_clear_button():
    """Add a clear button to reset the app state."""
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Clear 🗑️"):
            keys_to_clear = ["ocr_result", "uploaded_file", "processed_image"]
            for model_id in MODEL_CONFIGS:
                result_key = f"{model_id}_result"
                keys_to_clear.append(result_key)

            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

            # Force file uploader to reset
            st.session_state["file_uploader_key"] = str(time.time())
            st.rerun()


def process_uploaded_image(image):
    """Process and display the uploaded image."""
    img_width, img_height = image.size
    max_width, max_height = 600, 500

    # Calculate scaling factor to maintain aspect ratio
    width_ratio = max_width / img_width
    height_ratio = max_height / img_height
    scale_factor = min(width_ratio, height_ratio)

    # Only resize if needed
    if img_width > max_width or img_height > max_height:
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        image = image.resize((new_width, new_height))

    # Store the processed image in session state
    st.session_state["processed_image"] = image
    st.image(image, caption="Uploaded Image", use_container_width=False)
    return image


def check_for_error(result):
    """Check if the OCR model has an error."""
    if "error" in result:
        return True

    raw_text = result.get("raw_text", "")
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    return raw_text.startswith("Error:") or (
        "success" in result and result["success"] is False
    )


def has_no_text(result):
    """Check if the OCR model has no text."""
    raw_text = result.get("raw_text", "")
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    return raw_text == "No text found"


def display_result(label, result, proc_time, model_id, model_logos):
    """Display the results of the OCR model."""
    model_config = MODEL_CONFIGS[model_id]
    model_display = model_config.display

    # Display header with logo if available
    if model_display in model_logos:
        header_html = f'<div style="display: flex; align-items: center; margin-bottom: 10px;"><img src="data:image/svg+xml;base64,{model_logos[model_display]}" width="30" style="margin-right: 10px;"> <h3 style="margin: 0;">{label} Results</h3></div>'
        st.markdown(header_html, unsafe_allow_html=True)
    else:
        st.subheader(f"{label} Results")

    st.text(f"Processing time: {proc_time:.2f}s")
    st.markdown("##### Extracted Text")

    text = result.get("raw_text", "")
    if not isinstance(text, str):
        text = str(text)

    if text.startswith("Error:"):
        st.error(text)
    elif text == "No text found":
        st.warning("No text found in the image")
    else:
        st.write(text)


def run_single_model(image, model_id, custom_prompt):
    """Run OCR with a single model and display results."""
    selected_model_display = MODEL_CONFIGS[model_id].display
    model_logos = load_model_logos()

    with st.spinner(f"Processing image with {selected_model_display}..."):
        try:
            start = time.time()

            # Use the unified run_ocr function
            result = run_ocr(
                image_input=image,
                model_ids=model_id,
                custom_prompt=custom_prompt,
                track_metadata=False,
            )

            proc_time = time.time() - start

            # Store result in session state
            st.session_state[f"{model_id}_result"] = result

            # Display results
            display_result(
                selected_model_display,
                result,
                proc_time,
                model_id,
                model_logos,
            )

            # Show additional stats
            st.subheader("Processing Stats")
            st.text(f"Processing time: {proc_time:.2f}s")
            st.text(f"Text length: {len(result['raw_text'])} characters")

            if "confidence" in result and result["confidence"] is not None:
                st.text(f"Confidence: {result['confidence']:.2%}")

            st.text(
                f"Provider: {MODEL_CONFIGS[model_id].ocr_processor.capitalize()}"
            )

            return result, proc_time

        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None, 0


def display_comparison_stats(
    model_ids, model_results, model_times, model_errors
):
    """Display comparison statistics for multiple models."""
    st.markdown("### Comparison Stats")

    # Determine fastest model
    valid_times = []
    for model_id in model_ids:
        if not model_errors[model_id]:
            valid_times.append(
                (MODEL_CONFIGS[model_id].display, model_times[model_id])
            )

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
        if not model_errors[model_id] and not has_no_text(
            model_results[model_id]
        ):
            text_lengths.append(
                f"{MODEL_CONFIGS[model_id].display}: {len(model_results[model_id]['raw_text'])} chars"
            )

    if text_lengths:
        st.write("📝 Text lengths: " + ", ".join(text_lengths))
    else:
        st.warning("No usable text extracted by any model")


def run_multiple_models(image, model_ids, custom_prompt):
    """Run OCR with multiple models and display comparison results in parallel."""
    try:
        model_logos = load_model_logos()
        num_models = len(model_ids)

        # Create responsive layout
        if num_models <= 2:
            cols = st.columns(num_models)
            col_rows = None
        else:
            cols = None
            col_rows = []
            for i in range(0, num_models, 3):
                remaining = min(3, num_models - i)
                col_rows.append(st.columns(remaining))

        # Create placeholders with headers and spinners
        placeholders = {}
        for i, model_id in enumerate(model_ids):
            model_config = MODEL_CONFIGS[model_id]
            model_name = model_config.display

            # Get appropriate column
            if num_models <= 2:
                column = cols[i]
            else:
                row_idx = i // 3
                col_idx = i % 3
                column = col_rows[row_idx][col_idx]

            with column:
                # Add model header with logo
                if model_name in model_logos:
                    header_html = f'<div style="display: flex; align-items: center; margin-bottom: 10px;"><img src="data:image/svg+xml;base64,{model_logos[model_name]}" width="30" style="margin-right: 10px;"> <h3 style="margin: 0;">{model_name}</h3></div>'
                    st.markdown(header_html, unsafe_allow_html=True)
                else:
                    st.subheader(f"{model_name}")

                # Create spinner that will show during processing
                spinner_placeholder = st.empty()
                with spinner_placeholder:
                    st.info("Waiting to process...")

                # Create result area placeholder
                result_area = st.empty()

                placeholders[model_id] = {
                    "column": column,
                    "spinner": spinner_placeholder,
                    "result_area": result_area,
                }

        # Initialize result tracking
        model_results = {}
        model_times = {}
        model_errors = {}
        processed_count = 0

        # Create a placeholder for comparison stats
        stats_container = st.container()

        # Define callback function to update UI as each model completes
        def process_callback(model_id, result):
            nonlocal processed_count
            proc_time = result.get("processing_time", 0)

            # Store results
            model_results[model_id] = result
            model_times[model_id] = proc_time
            model_errors[model_id] = check_for_error(result)

            # Store in session state
            st.session_state[f"{model_id}_result"] = result

            # Clear the spinner
            placeholders[model_id]["spinner"].empty()

            # Update the result in the UI (without the header which is already shown)
            with placeholders[model_id]["result_area"].container():
                text = result.get("raw_text", "")
                if not isinstance(text, str):
                    text = str(text)

                st.text(f"Processing time: {proc_time:.2f}s")
                st.markdown("##### Extracted Text")

                if text.startswith("Error:"):
                    st.error(text)
                elif text == "No text found":
                    st.warning("No text found in the image")
                else:
                    st.write(text)

            processed_count += 1

            # When all models are done, show comparison stats
            if processed_count == len(model_ids) and len(model_ids) > 1:
                with stats_container:
                    display_comparison_stats(
                        model_ids, model_results, model_times, model_errors
                    )

        # Process each model in a separate thread
        from concurrent.futures import ThreadPoolExecutor

        def process_model(model_id):
            try:
                # Don't update UI from the thread - just log it
                start_time = time.time()
                result = run_ocr(
                    image_input=image,
                    model_ids=model_id,
                    custom_prompt=custom_prompt,
                    track_metadata=False,
                )
                # Ensure processing time is captured
                if "processing_time" not in result:
                    result["processing_time"] = time.time() - start_time
                return model_id, result
            except Exception as e:
                error_result = {
                    "raw_text": f"Error: {str(e)}",
                    "error": str(e),
                    "processing_time": 0,
                    "model": model_id,
                }
                return model_id, error_result

        # Use max_workers based on number of models
        max_workers = min(len(model_ids), 5)

        # Create a dict to track which models are currently processing
        processing_models = {model_id: False for model_id in model_ids}

        # Start processing in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Update all spinners to "Processing..." before submitting tasks
            for model_id in model_ids:
                with placeholders[model_id]["spinner"]:
                    st.info(
                        f"Processing with {MODEL_CONFIGS[model_id].display}..."
                    )
                processing_models[model_id] = True

            # Submit tasks
            futures = {
                executor.submit(process_model, model_id): model_id
                for model_id in model_ids
            }

            # Process results as they complete
            import concurrent.futures

            for future in concurrent.futures.as_completed(futures):
                model_id, result = future.result()
                process_callback(model_id, result)

        return model_results, model_times

    except Exception as e:
        st.error(f"Error processing images: {e}")
        return None, None


def main():
    """Main function to run the Streamlit app."""
    setup_page_config()
    uploaded_file, custom_prompt = render_sidebar()
    model_logos = load_model_logos()
    render_header(model_logos)
    add_clear_button()

    st.markdown("---")
    st.markdown(
        '<p style="margin-top: -20px;">Extract structured text from images using your chosen OCR model!</p>',
        unsafe_allow_html=True,
    )

    processed_image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        processed_image = process_uploaded_image(image)
    elif "processed_image" in st.session_state:
        processed_image = st.session_state["processed_image"]
        st.image(
            processed_image,
            caption="Uploaded Image",
            use_container_width=False,
        )

    if processed_image is not None:
        if st.button("Extract Text 🔍", type="primary"):
            comparison_mode = st.session_state.get("comparison_mode", False)

            if comparison_mode:
                comparison_models = st.session_state.get(
                    "comparison_models", [DEFAULT_MODEL.name]
                )
                run_multiple_models(
                    processed_image, comparison_models, custom_prompt
                )
            else:
                selected_model_id = st.session_state.get(
                    "selected_model_id", DEFAULT_MODEL.name
                )
                run_single_model(
                    processed_image, selected_model_id, custom_prompt
                )
    else:
        st.info(
            "Upload an image and click 'Extract Text' to see the results here."
        )

    # Footer
    st.markdown("---")
    st.caption("ZenOCR - Comparing LLM OCR capabilities")


if __name__ == "__main__":
    main()
