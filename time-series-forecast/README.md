# 🍃 :arrow_right: 🔌 Predict electricity power generation based on wind forecast in Orkney, Scotland 

By its nature, renewable energy is highly weather-dependent, and the ongoing expansion of renewables is making our global power supply more vulnerable to changing weather conditions. To be able to predict how much power will be generated based on the weather forecast might be very crucial especially for areas such as Orkney in Scotland.


In this repository I showcase how to:
- Build a retrainable `zenml` pipeline
- Load data from Google Cloud Bigquery as a part of a `zenml` pipeline
- In the pipeline train your model remotely in Google Cloud Vertex AI

## 🗺 Overview

<HOW THIS INTERFACES WITH ZENML>

## 🧰 How the example is implemented

<WHAT SHOULD THE USER EXPECT WHEN GOING THROUGH THE EXAMPLE>

<HIGHLIGHT INTERESTING CODE SNIPPETS>

<OPTIONALLY SHOW INTERESTING GRAPHICS OF WHAT OUTPUT TO EXPECT>

<IN CASE OF HIGH COMPLEXITY: AN ARCHITECTURE DIAGRAM>

# 🖥 Run it locally

## ⏩ SuperQuick `<EXAMPLE_NAME>` run

<ONLY ADD THIS SECTION IF A setup.sh FILE IS INCLUDED IN THE EXAMPLE>

If you're really in a hurry and just want to see this example pipeline run
without wanting to fiddle around with all the individual installation and
configuration steps, just run the following:

```shell
zenml example run <NAME OF EXAMPLE>
```

## 👣 Step-by-Step

### 📄 Prerequisites

In order to run this example, you need to install and initialize ZenML:

```shell
# install CLI
pip install zenml

# install ZenML integrations
zenml integration install <NAMES OF THE INTEGRATIONS>

# pull example
zenml example pull <NAME OF THE EXAMPLE>
cd zenml_examples/<NAME OF THE EXAMPLE>

# Initialize ZenML repo
zenml init
```

### ▶️ Run the Code

Now we're ready. Execute:

```bash
python run.py
```

### 🧽 Clean up

In order to clean up, delete the remaining ZenML references.

```shell
rm -rf zenml_examples
```

# 📜 Learn more

Our docs regarding the <NAME OF THE INTEGRATION> integration can be
found [here](<LINK TO THE DOCS>).

If you want to learn more about <EXAMPLE_TYPE> in general or about how to build
your own <TYPE OF THE INTEGRATION> in ZenML
check out our [docs](<LINK TO THE DOCS>).
