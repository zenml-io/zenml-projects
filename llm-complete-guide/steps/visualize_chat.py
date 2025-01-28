from typing import Any, Dict

from typing_extensions import Annotated
from zenml import get_step_context, log_metadata, step
from zenml.metadata.metadata_types import Uri
from zenml.types import HTMLString
from zenml.utils.dashboard_utils import get_model_version_url


@step(enable_cache=False)
def create_chat_interface(
    deployment_info: Dict[str, Any],
) -> Annotated[HTMLString, "chat_bot"]:
    step_context = get_step_context()
    html = """
    <div id="zenml-chat-container" class="w-full max-w-4xl mx-auto">
        <style>
            #zenml-chat-container {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            #zenml-chat-window {
                background: white;
                border-radius: 0.5rem;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                display: flex;
                flex-direction: column;
                height: 600px;
                width: 100%;
                overflow: hidden;
            }
            
            #zenml-chat-header {
                padding: 1rem;
                background: #f8f9fa;
                border-bottom: 1px solid #eee;
            }
            
            #zenml-chat-header h1 {
                margin: 0;
                font-size: 1.25rem;
                color: #333;
            }
            
            #zenml-chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 1rem;
                min-height: 400px;
                max-height: calc(100% - 120px);
            }
            
            .zenml-message {
                margin-bottom: 1rem;
                display: flex;
                align-items: flex-start;
            }
            
            .zenml-message.user {
                flex-direction: row-reverse;
            }
            
            .zenml-message-content {
                max-width: 80%;
                padding: 0.75rem 1rem;
                border-radius: 1rem;
                background: #f0f2f5;
                margin: 0 0.75rem;
                word-wrap: break-word;
                overflow-wrap: break-word;
            }
            
            .zenml-message.user .zenml-message-content {
                background: #0084ff;
                color: white;
            }
            
            #zenml-chat-input {
                padding: 1rem;
                border-top: 1px solid #eee;
                background: white;
                display: flex;
                gap: 0.625rem;
            }
            
            #zenml-user-input {
                flex: 1;
                padding: 0.75rem;
                border: 1px solid #ddd;
                border-radius: 0.375rem;
                font-size: 0.875rem;
                min-width: 0;
            }
            
            #zenml-send-button {
                padding: 0.75rem 1.5rem;
                background: #0084ff;
                color: white;
                border: none;
                border-radius: 0.375rem;
                cursor: pointer;
                font-size: 0.875rem;
                white-space: nowrap;
            }
            
            #zenml-send-button:hover {
                background: #0073e6;
            }
            
            #zenml-send-button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            
            #zenml-typing-indicator {
                display: none;
                padding: 0.75rem 1rem;
                background: #f0f2f5;
                border-radius: 1rem;
                margin: 0 0.75rem 1rem;
                font-style: italic;
                color: #666;
            }
            
            .zenml-message-content pre {
                background: rgba(0, 0, 0, 0.05);
                padding: 0.5rem;
                border-radius: 0.25rem;
                overflow-x: auto;
                max-width: 100%;
            }
            
            .zenml-message-content code {
                font-family: monospace;
                font-size: 0.875em;
            }
            
            .zenml-message-content p {
                margin-bottom: 0.5rem;
            }
            
            .zenml-message-content p:last-child {
                margin-bottom: 0;
            }
            
            .zenml-message-content * {
                max-width: 100%;
                overflow-wrap: break-word;
                word-wrap: break-word;
                word-break: break-word;
            }
        </style>
        
         <div id="zenml-chat-window">
            <div id="zenml-chat-header">
                <h1>ZenML Assistant</h1>
            </div>
            
            <div id="zenml-chat-messages">
                <div id="zenml-typing-indicator">
                    Assistant is typing...
                </div>
            </div>
            
            <div id="zenml-chat-input">
                <input type="text" id="zenml-user-input" placeholder="Type your message..." />
                <button id="zenml-send-button">Send</button>
            </div>
        </div>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
        <script>
            // Initialize immediately instead of waiting for load event
            (function() {
                const STORAGE_KEY = 'zenmlChatHistory-' + window.location.pathname;
                const chatMessages = document.getElementById('zenml-chat-messages');
                const userInput = document.getElementById('zenml-user-input');
                const sendButton = document.getElementById('zenml-send-button');
                const typingIndicator = document.getElementById('zenml-typing-indicator');
                let messages = [];

                // Initialize chat history immediately
                function initializeChatHistory() {
                    try {
                        const storedMessages = localStorage.getItem(STORAGE_KEY);
                        if (storedMessages) {
                            messages = JSON.parse(storedMessages);
                            if (Array.isArray(messages) && messages.length > 0) {
                                messages.forEach(message => appendMessage(message.text, message.isUser, true));
                            } else {
                                addInitialMessage();
                            }
                        } else {
                            addInitialMessage();
                        }
                    } catch (error) {
                        console.error('Error loading chat history:', error);
                        addInitialMessage();
                    }
                }

                function addInitialMessage() {
                    const initialMessage = "Hi! I'm your ZenML assistant. How can I help you today?";
                    appendMessage(initialMessage, false);
                }

                function appendMessage(text, isUser, isLoading = false) {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `zenml-message ${isUser ? 'user' : ''}`;
                    
                    const contentDiv = document.createElement('div');
                    contentDiv.className = 'zenml-message-content';
                    
                    if (!isUser) {
                        contentDiv.innerHTML = marked.parse(text);
                    } else {
                        contentDiv.textContent = text;
                    }
                    
                    messageDiv.appendChild(contentDiv);
                    chatMessages.insertBefore(messageDiv, typingIndicator);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                    
                    if (!isLoading) {
                        messages.push({ text, isUser });
                        try {
                            localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
                        } catch (error) {
                            console.error('Error saving to localStorage:', error);
                        }
                    }
                }
                
                async function sendMessage() {
                    const message = userInput.value.trim();
                    if (!message) return;
                    
                    userInput.disabled = true;
                    sendButton.disabled = true;
                    
                    appendMessage(message, true);
                    userInput.value = '';
                    
                    typingIndicator.style.display = 'block';
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                    
                    try {
                        const response = await fetch('https://chat-rag.staging.cloudinfra.zenml.io/generate', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                query: message,
                                temperature: 0.4,
                                max_tokens: 1000
                            })
                        });
                        
                        if (!response.ok) {
                            throw new Error('Network response was not ok');
                        }
                        
                        const reader = response.body.getReader();
                        let assistantResponse = '';
                        
                        while (true) {
                            const {done, value} = await reader.read();
                            if (done) break;
                            
                            const text = new TextDecoder().decode(value);
                            assistantResponse += text;
                        }
                        
                        typingIndicator.style.display = 'none';
                        appendMessage(assistantResponse, false);
                    } catch (error) {
                        console.error('Error:', error);
                        typingIndicator.style.display = 'none';
                        appendMessage('Sorry, there was an error processing your request.', false);
                    }
                    
                    userInput.disabled = false;
                    sendButton.disabled = false;
                    userInput.focus();
                }
                
                userInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });
                
                sendButton.addEventListener('click', sendMessage);

                // Initialize immediately
                initializeChatHistory();
            })();
        </script>
    </div>
    """
    model_version_url = get_model_version_url(step_context.model.id)
    log_metadata(
        infer_artifact=True,
        metadata={
            "deployment_info": deployment_info,
            "deployment_url": Uri(f"{model_version_url}/?tab=deployments"),
        },
    )
    return HTMLString(html)
