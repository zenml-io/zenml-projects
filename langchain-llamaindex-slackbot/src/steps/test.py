from __future__ import annotations

PIPELINE_NAME = "zenml_agent_creation_pipeline"

PREFIX = """ZenML Agent is a large language model operated by ZenML. It can answer user's questions about ZenML for the right version
by first asking the user what version they are using. ZenML agent DOES NOT make up
answers by itself. it always uses the right tools available, given the context. NEVER make up answers.

It is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a large number of topics, including about ZenML.
When a user asks a question pertaining to ZenML, *YOU MUST* ask the user for the version of ZenML that they are using and 
then use a tool corresponding to the right version to answer questions based on that.

It is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, ZenML Agent is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, ZenML Agent is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, ZenML Agent is here to assist."""

SUFFIX = """TOOLS
------
ZenML Agent can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:

{{tools}}

{format_instructions}

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

{{{{input}}}}"""

FORMAT_INSTRUCTIONS = """RESPONSE FORMAT INSTRUCTIONS
----------------------------

When responding to me, please output a response in one of two formats:

**Option 1:**
Use this if you want to use a tool to answer a question. For example, if you have already
asked the user/human for the version they are using, you can then use the corresponding tool
to answer their question.
Markdown code snippet formatted in the following schema:

```json
{{{{
    "action": string, \\ The action to take. Must be one of {tool_names}
    "action_input": string \\ The input to the action
}}}}
```

**Option #2:**
Use this when you have to ask the user about the ZenML version. Or for other general direct chat with the human. Markdown code snippet formatted in the following schema:

```json
{{{{
    "action": "Final Answer",
    "action_input": string \\ You should put what you want to return to user here
}}}}
```"""

from zenml.client import Client
from langchain.chat_models import ChatOpenAI

from langchain.agents import ConversationalChatAgent

from typing import List, Optional, Sequence


from langchain.tools.base import BaseTool

from langchain.schema import (
    BaseOutputParser,
)
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.schema import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools.vectorstore.tool import VectorStoreQATool
from langchain.memory.vectorstore import VectorStoreRetrieverMemory
from langchain.agents import AgentExecutor


from typing import Union

from langchain.agents import AgentOutputParser
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish, OutputParserException


class ZenMLAgent(ConversationalChatAgent):
    """ZenML flavored Conversational Chat Agent."""

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        system_message: str = PREFIX,
        human_message: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> BasePromptTemplate:
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        _output_parser = output_parser or cls._get_default_output_parser()
        format_instructions = human_message.format(
            format_instructions=_output_parser.get_format_instructions()
        )
        final_prompt = format_instructions.format(
            tool_names=tool_names, tools=tool_strings
        )
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(final_prompt),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        return ChatPromptTemplate(
            input_variables=input_variables, messages=messages
        )


# class ZenMLOutputParser(AgentOutputParser):
#     """Output parser for the conversational agent."""

#     def get_format_instructions(self) -> str:
#         """Returns formatting instructions for the given output parser."""
#         return FORMAT_INSTRUCTIONS

#     def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
#         """Attempts to parse the given text into an AgentAction or AgentFinish.

#         Raises:
#              OutputParserException if parsing fails.
#         """
#         try:
#             # Attempt to parse the text into a structured format (assumed to be JSON
#             # stored as markdown)
#             # if text is a json, parse it otherwise assign directly
#             if text.startswith("{"):
#                 response = parse_json_markdown(text)
#             else:
#                 response = {"action": "Final Answer", "action_input": text}

#             # If the response contains an 'action' and 'action_input'
#             if "action" in response and "action_input" in response:
#                 action, action_input = (
#                     response["action"],
#                     response["action_input"],
#                 )

#                 # If the action indicates a final answer, return an AgentFinish
#                 if action == "Final Answer":
#                     return AgentFinish({"output": action_input}, text)
#                 else:
#                     # Otherwise, return an AgentAction with the specified action and
#                     # input
#                     return AgentAction(action, action_input, text)
#             else:
#                 # If the necessary keys aren't present in the response, raise an
#                 # exception
#                 raise OutputParserException(
#                     f"Missing 'action' or 'action_input' in LLM output: {text}"
#                 )
#         except Exception as e:
#             breakpoint()
#             # If any other exception is raised during parsing, also raise an
#             # OutputParserException
#             raise OutputParserException(
#                 f"Could not parse LLM output: {text}"
#             ) from e

#     @property
#     def _type(self) -> str:
#         return "conversational_chat"


if __name__ == "__main__":
    pipeline_model9 = Client().get_pipeline(
        name_id_or_prefix=PIPELINE_NAME, version="9"
    )

    # get the pipeline version as an int
    pipeline_version = int(pipeline_model9.version)

    # get the latest run of the previous version

    version = "0.44.1"

    existing_tools = []
    llm = {
        "temperature": 0,
        "max_tokens": 1000,
        "model_name": "gpt-3.5-turbo",
    }

    if pipeline_model9.runs is not None:
        # get the last run
        last_run = pipeline_model9.runs[0]
        # get the agent_creator step
        agent_creator_step = last_run.steps["agent_creator"]

        vector_store = last_run.steps["index_generator"].output.load()
        # get the output
        try:
            existing_agent = agent_creator_step.output.load()

            existing_tools = existing_agent.tools
        except ValueError:
            print("No existing agent found.")

        print(vector_store)
        print(type(vector_store))

        tools = [
            *existing_tools,
            VectorStoreQATool(
                name=f"zenml-{version}",
                vectorstore=vector_store,
                description="Use this tool to answer questions about ZenML version "
                f"{version}. How to debug errors in ZenML, how to answer conceptual "
                "questions about ZenML like available features, existing abstractions, "
                "and other parts from the documentation.",
                llm=ChatOpenAI(**llm),
            ),
        ]

        embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
        index = faiss.IndexFlatL2(embedding_size)
        embedding_fn = OpenAIEmbeddings().embed_query
        memory_vectorstore = FAISS(
            embedding_fn, index, InMemoryDocstore({}), {}
        )

        memory = VectorStoreRetrieverMemory(
            retriever=memory_vectorstore.as_retriever(),
            memory_key="chat_history",
        )

        my_agent = ZenMLAgent.from_llm_and_tools(
            llm=ChatOpenAI(**llm),
            tools=tools,
            system_message=PREFIX,
            human_message=SUFFIX,
            # output_parser=ZenMLOutputParser(),
        )

        chat_history = []
        chat_history.append(
            HumanMessage(content="Hi, can you help me with zenml?")
        )
        chat_history.append(
            AIMessage(content="I sure can help you answer zenml questions")
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=my_agent,
            tools=tools,
            verbose=True,
        )

        breakpoint()
        agent_executor.run({"input": "can zenml work with mlflow? if yes, then how?","chat_history": chat_history})

        # TODO
        """
        - in the prompt, tell the model to ask for version the user is using
        - prompt: answer in detail
        """


"""
have a prompt template that incorporates good practices.
only take name, personality as input.
agent = Agent(name="ZenML Agent", personality="ZenML Agent is a large language model operated by ZenML. It smart af")

helper function called educate_agent that can take in a list of URLs,
and then create vector DB tools (or whatever other latest tech) based on different
versions that you can find in there. and then adds them to the agent.
this function should check for version numbers in the URLs and then create the tools
based on that + search for all versions available. to make the comparison easier, we can
take a latest version to know how a version looks like for the user.

agent.educate_agent(url=[["https://docs.zenml.io/", "0.44.1", skip_version=0.43.1], ["https://youtu.be/1yFwWx0mY5I"]], enable_sme=True) 
all general info like youtube and rest would by default go to the latest version
collection. if the version passed is new, then we would run pipeline on all the general info and
add to the latest version's collection. otherwise, we'll skip that step.


PIPELINE:
one docs pipeline. that is similar to the one now but returns list of things instead. so
list of vectir stores, list of list of URLs and so on.
when the hleoer function is called, we first check the alreayd available list of tools from
the latest agent. if there are tools that are missing (each tool should be given a unique name)
we only run the pipeline for those tools. otherwise we skip it.
for example, if a new version is specified in the function, we see what the last version
is in our toolkit and we run the pipeline for all the versions excluding the skip_version
version.

has the ability to summarize contents into tool descriptions with date of creation.
fn to add urls later too.


Tool definition

VectorStoreTOOL(
    
    vector_store=eduaction_handler.get(URL) or just a vector store.
    unknown_policy=UnknownPolicy.ASK_FOR_SME,
    )
handler might not be the most convenient idea. by default if enable sme is true then the
helper function sets the sme unknown policy. otherwise it's ignore.
    
.deplpy first compiles the agent. the compilation step will then take in all the handler
arguments and build the zenml pipeline. this would involve using a step 

agent.check_status() or something
# call happens in the background ^
calls a zenml pipeline that creates the urls, loads them and then creates the tools
caching by default. separate pipeline for each urL??
if a new URL gets added, don't run the whole pipeline again. dynamic steps new steps 
for each loader that all converge in one index generator step that checks for old
indexes and updates them.

have a deploy agent function that deploys the agent as an endpoint. allow version switching
should come with a default memory that gets refreshed for every session. thread specific memory.
also comes with a validator by default that stops offensive messages. use the library alex sent.

agent.deploy(version="0.44.1", memory=Memory(), validator=Validator())
test_agent = agent.deploy(version="0.44.1", memory=Memory(), validator=Validator(), test=True)

test_agent.ask("can zenml work with mlflow? if yes, then how?")

have a type to each tool that the llm knows about.
when no version is provided in the question, use the latest. only provide other version
info when asked about it. in a previous version if sth was different, provide that too.

can have a collection called experts and then an action that can decide to query for this db
to get the most knowledgeable person on the topic. then ping them.


if the output of a knowledge type tool is "unknown", then prompt the LLM in the next cycle
to figure out the subject matter expert. this should be part of an unknown policy. maybe
attach it to every tool.
UnkwonPolicy.ASK_FOR_SME, UnknownPolicy.IGNORE
the policy could find the subject matter expert and ping them, or it could outout the name to
the human. the first case would do it silently (without showing anything to user maybe) for 
companies that want to set up a feedback mechanism.
there has to be a message (or get_human_input) action, not just tool and final answer.

my agent will have this extra action and every iteration of a loop will send the intermediate
steps no doubt. if it's too big, consider summarizing it.

tool = DocSearcher() # use the best retriever tech, if vector stores then that otherwise the other paper.

an agent has a model and a set of tools
make it compatible with langchain tools
implement some of your own tools using the BaseTool abstract class
one of them would be the doc searcher tool with the latest paper

basically make your basetool abstraction the same as the langchain one and
you can then take input a langchain tool and use it in your agent.


a native cost tracking mechanism.
a way to train custom LLMs on the input URLs for open source models. just trigger a training
pipeline in the background.

the cloud will have options to run the backing pipeline remotely,
deploy remotely. set up triggers for new versions and more.

"""