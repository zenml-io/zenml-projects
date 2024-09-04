# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
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
#
import os
import shutil
import signal
import subprocess
from textwrap import dedent

from zenml.logger import get_logger

logger = get_logger(__name__)


def generate_importance_matrix(model_path, train_data_path):
    """Generate importance matrix for the model using llama-imatrix.
    
    Args:
        model_path (str): Path to the model file.
        train_data_path (str): Path to the training data file.
    """
    imatrix_command = f"./llama-imatrix -m {model_path} -f {train_data_path} -ngl 99 --output-frequency 10"

    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in the current directory: {os.listdir('.')}")

    if not os.path.isfile(model_path):
        raise Exception(f"Model file not found: {model_path}")

    logger.info("Running imatrix command...")
    process = subprocess.Popen(imatrix_command, shell=True)

    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        logger.info("Imatrix computation timed out. Sending SIGINT to allow graceful termination...")
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.info("Imatrix proc still didn't term. Forcefully terming process...")
            process.kill()

    logger.info("Importance matrix generation completed.")

def split_model(model_path, output_dir, split_max_tensors=256, split_max_size=None):
    """Split the model into multiple parts using llama-gguf-split.
    
    Args:
        model_path (str): Path to the model file.
        output_dir (str): Path to the output directory.
        split_max_tensors (int): Maximum number of tensors per split.
        split_max_size (int): Maximum size of each split.
    """
    split_cmd = f"./llama-gguf-split --split --split-max-tensors {split_max_tensors}"
    if split_max_size:
        split_cmd += f" --split-max-size {split_max_size}"
    split_cmd += f" {model_path} {os.path.join(output_dir, os.path.basename(model_path).split('.')[0])}"

    logger.info(f"Split command: {split_cmd}")

    result = subprocess.run(split_cmd, shell=True, capture_output=True, text=True)
    logger.info(f"Split command stdout: {result.stdout}")
    logger.info(f"Split command stderr: {result.stderr}")

    if result.returncode != 0:
        raise Exception(f"Error splitting the model: {result.stderr}")
    logger.info("Model split successfully!")

def process_model(
    model_path, 
    output_dir, 
    model_name,
    q_method="Q4_K_M", 
    use_imatrix=False, 
    imatrix_q_method="IQ4_NL",
    train_data_file=None, 
    split_model=False, 
    split_max_tensors=256, 
    split_max_size=None
):
    """Process the model using llama.cpp.
    
    Args:
        model_path (str): Path to the model file.
        output_dir (str): Path to the output directory.
        q_method (str): Quantization method to use.
        use_imatrix (bool): Whether to use importance matrix quantization.
        imatrix_q_method (str): Importance matrix quantization method to use.
        train_data_file (str): Path to the training data file.
        split_model (bool): Whether to split the model.
        split_max_tensors (int): Maximum number of tensors per split.
        split_max_size (int): Maximum size of each split.
    """
    fp16 = os.path.join(output_dir, f"{model_name}.fp16.gguf")

    try:
        os.makedirs(output_dir, exist_ok=True)

        conversion_script = "./convert_hf_to_gguf.py"
        fp16_conversion = f"python {conversion_script} {model_path} --outtype f16 --outfile {fp16}"
        result = subprocess.run(fp16_conversion, shell=True, capture_output=True)
        logger.info(result)
        if result.returncode != 0:
            raise Exception(f"Error converting to fp16: {result.stderr}")
        logger.info("Model converted to fp16 successfully!")
        logger.info(f"Converted model path: {fp16}")

        imatrix_path = "./imatrix.dat"

        if use_imatrix:
            if train_data_file:
                train_data_path = train_data_file
            else:
                train_data_path = "groups_merged.txt"  # fallback calibration dataset

            logger.info(f"Training data file path: {train_data_path}")

            if not os.path.isfile(train_data_path):
                raise Exception(f"Training data file not found: {train_data_path}")

            generate_importance_matrix(fp16, train_data_path)
        else:
            logger.info("Not using imatrix quantization.")

        quantized_gguf_name = f"{model_name.lower()}-{imatrix_q_method.lower()}-imat.gguf" if use_imatrix else f"{model_name.lower()}-{q_method.lower()}.gguf"
        quantized_gguf_path = os.path.join(output_dir, quantized_gguf_name)

        if use_imatrix:
            quantise_ggml = f"./llama-quantize --imatrix {imatrix_path} {fp16} {quantized_gguf_path} {imatrix_q_method}"
        else:
            quantise_ggml = f"./llama-quantize {fp16} {quantized_gguf_path} {q_method}"

        result = subprocess.run(quantise_ggml, shell=True, capture_output=True)
        if result.returncode != 0:
            raise Exception(f"Error quantizing: {result.stderr}")
        logger.info(f"Quantized successfully with {imatrix_q_method if use_imatrix else q_method} option!")
        logger.info(f"Quantized model path: {quantized_gguf_path}")

        readme_content = dedent(
            f"""
            # {model_name}-{imatrix_q_method if use_imatrix else q_method}-GGUF
            This model was converted to GGUF format from the original model using llama.cpp.

            ## Use with llama.cpp
            Install llama.cpp through brew (works on Mac and Linux)

            ```bash
            brew install llama.cpp
            ```

            Invoke the llama.cpp server or the CLI.

            ### CLI:
            ```bash
            llama-cli -m 0cb84d17-q4_k_m_1.gguf -p "The meaning to life and the universe is"
            ```

            ### Server:
            ```bash
            llama-server -m {quantized_gguf_name} -c 2048
            ```
            """
        )

        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(readme_content)

        if split_model:
            split_model(quantized_gguf_path, output_dir, split_max_tensors, split_max_size)

        if os.path.isfile(imatrix_path):
            shutil.copy(imatrix_path, os.path.join(output_dir, "imatrix.dat"))

        logger.info(f'Model processed successfully. Output saved to: {output_dir}')
    except Exception as e:
        logger.info(f"Error: {e}")