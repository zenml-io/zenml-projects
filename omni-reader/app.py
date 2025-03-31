# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This is the Streamlit UI for the ZenML OCR Extraction workflow."""

import base64
import time

import streamlit as st
from PIL import Image

from run_compare_ocr import run_ocr_from_ui, run_ollama_ocr_from_ui

# Page configuration
st.set_page_config(
    page_title="OCR Extraction",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Helper functions
def get_model_param(selected_model):
    """Get the model parameter for the OCR model."""
    mapping = {"Mistral Pixtral": "pixtral-12b-2409", "Gemma-3": "gemma3:27b"}
    return mapping.get(selected_model, "both")


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


# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Select OCR Model",
        options=["Gemma-3", "Mistral Pixtral", "Compare Both"],
    )
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
    st.caption("Powered by Ollama + MistralAI")

gemma_logo = base64.b64encode(open("./assets/logos/gemma.svg", "rb").read()).decode()
mistral_logo = base64.b64encode(open("./assets/logos/mistral.svg", "rb").read()).decode()

# Main header with optional logo
if model_choice == "Gemma-3":
    st.markdown(
        f'# <img src="data:image/svg+xml;base64,{gemma_logo}" width="50" style="vertical-align: -12px;"> Gemma-3 OCR',
        unsafe_allow_html=True,
    )
elif model_choice == "Mistral Pixtral":
    st.markdown(
        f'# <img src="data:image/svg+xml;base64,{mistral_logo}" width="50" style="vertical-align: -4px;"> Pixtral 12B OCR',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
        f'<img src="data:image/svg+xml;base64,{gemma_logo}" width="50" style="vertical-align: middle;">'
        f'<h1 style="margin: 0 15px;">OCR Model Comparison</h1>'
        f'<img src="data:image/svg+xml;base64,{mistral_logo}" width="50" style="vertical-align: middle;">'
        f"</div>",
        unsafe_allow_html=True,
    )

# Clear button
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Clear 🗑️"):
        keys_to_clear = ["ocr_result", "gemma_result", "mistral_result"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.session_state["uploaded_file"] = None

        # This forces the file uploader to reset on next rerun
        st.session_state["file_uploader_key"] = str(time.time())  # Changes the key each time

        st.rerun()

st.markdown("---")
st.markdown(
    '<p style="margin-top: -20px;">Extract structured text from images using your chosen OCR model!</p>',
    unsafe_allow_html=True,
)

# Process uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)

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

    if st.button("Extract Text 🔍", type="primary"):
        prompt_param = custom_prompt if use_custom_prompt else None
        model_param = get_model_param(model_choice)

        if model_param == "both":
            with st.spinner("Processing image with both models..."):
                try:
                    col1, col2 = st.columns(2)

                    start = time.time()
                    # gemma_result = run_ocr_from_ui(
                    #     image=image, model="ollama/gemma3:27b", custom_prompt=prompt_param
                    # )
                    gemma_result = run_ollama_ocr_from_ui(
                        image,
                        model="gemma3:27b",
                        custom_prompt=prompt_param,
                    )
                    gemma_time = time.time() - start

                    start = time.time()
                    mistral_result = run_ocr_from_ui(
                        image=image,
                        model="pixtral-12b-2409",
                        custom_prompt=prompt_param,
                    )
                    mistral_time = time.time() - start

                    with col1:
                        display_result("Gemma-3", gemma_result, gemma_time)
                    with col2:
                        display_result("Pixtral 12B", mistral_result, mistral_time)

                    st.session_state["gemma_result"] = gemma_result
                    st.session_state["mistral_result"] = mistral_result

                    gemma_error, mistral_error = (
                        check_for_error(gemma_result),
                        check_for_error(mistral_result),
                    )
                    st.markdown("### Comparison Stats")
                    if not gemma_error and not mistral_error:
                        faster = "Gemma-3" if gemma_time < mistral_time else "Mistral Pixtral"
                        st.write(
                            f"🚀 Faster model: **{faster}** (by {abs(gemma_time - mistral_time):.2f}s)"
                        )
                    elif gemma_error and not mistral_error:
                        st.write(
                            f"⚠️ Gemma-3 failed, Mistral Pixtral completed in {mistral_time:.2f}s"
                        )
                    elif mistral_error and not gemma_error:
                        st.write(
                            f"⚠️ Mistral Pixtral failed, Gemma-3 completed in {gemma_time:.2f}s"
                        )
                    else:
                        st.error("Both models failed to process the image")

                    # Compare extracted text lengths
                    if not gemma_error and not mistral_error:
                        gemma_text = gemma_result["raw_text"]
                        mistral_text = mistral_result["raw_text"]
                        if has_no_text(gemma_result) and has_no_text(mistral_result):
                            st.warning("Neither model found any text in the image")
                        elif has_no_text(gemma_result):
                            st.write(
                                f"📝 No text found by Gemma-3, Mistral found {len(mistral_text)} chars"
                            )
                        elif has_no_text(mistral_result):
                            st.write(
                                f"📝 No text found by Mistral, Gemma-3 found {len(gemma_text)} chars"
                            )
                        else:
                            st.write(
                                f"📝 Text length: Gemma-3: {len(gemma_text)} chars, Mistral: {len(mistral_text)} chars"
                            )
                    else:
                        if not gemma_error:
                            st.write(
                                "📝 Gemma-3 "
                                + (
                                    "found no text"
                                    if has_no_text(gemma_result)
                                    else f"extracted {len(gemma_result['raw_text'])} chars"
                                )
                            )
                        if not mistral_error:
                            st.write(
                                "📝 Mistral Pixtral "
                                + (
                                    "found no text"
                                    if has_no_text(mistral_result)
                                    else f"extracted {len(mistral_result['raw_text'])} chars"
                                )
                            )
                except Exception as e:
                    st.error(f"Error processing image: {e}")
        else:
            with st.spinner(f"Processing image with {model_choice}..."):
                try:
                    start = time.time()
                    if "gemma" in model_param.lower():
                        response = run_ollama_ocr_from_ui(
                            image, model="gemma3:27b", custom_prompt=prompt_param
                        )
                    else:
                        response = run_ocr_from_ui(
                            image=image, model=model_param, custom_prompt=prompt_param
                        )
                    proc_time = time.time() - start
                    st.session_state["ocr_result"] = response

                    st.subheader("Extracted Text")
                    text = response["raw_text"]
                    if text.startswith("Error:"):
                        st.error(response)
                    elif text == "No text found":
                        st.warning("No text found in the image")
                    else:
                        st.write(text)

                    st.subheader("Processing Stats")
                    st.text(f"Processing time: {proc_time:.2f}s")
                    st.text(f"Text length: {len(response['raw_text'])} characters")
                except Exception as e:
                    st.error(f"Error processing image: {e}")
else:
    st.info("Upload an image and click 'Extract Text' to see the results here.")

# Footer
st.markdown("---")
st.caption("ZenOCR - Comparing LLM OCR capabilities")
