import streamlit as st
import requests
import json
import os
import datetime
import random
import string

# Alternative function to generate a unique ID without using uuid
def generate_unique_id(length=12):
    """Generate a random ID string"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def show_chat_ai():
    """Display the chat with AI page content"""
    
    # Apply chat AI specific styling
    st.markdown("""
        <style>
        /* Chat styling */
        .chat-container {
            margin-bottom: 20px;
            max-height: 600px;
            overflow-y: auto;
        }
        
        .user-message {
            background-color: #f0f2f6;
            border-radius: 18px;
            padding: 10px 15px;
            margin: 5px 0;
            margin-left: 40px;
            margin-right: 20px;
            text-align: right;
        }
        
        .assistant-message {
            background-color: #f8fbff;
            border: 1px solid #e6e9f0;
            border-radius: 8px;
            padding: 10px 15px;
            margin: 5px 0;
            margin-right: 40px;
        }
        
        .system-message {
            background-color: #f5f7fa;
            border-radius: 4px;
            padding: 5px 10px;
            margin: 5px 0;
            text-align: center;
            color: #666;
            font-size: 12px;
        }
        
        .message-metadata {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        
        /* Usage metrics panel */
        .metrics-panel {
            background-color: #f8f9fa;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
            display: flex;
            justify-content: space-between;
        }
        
        .metric-item {
            text-align: center;
        }
        
        /* Document selection */
        .document-selector {
            background-color: #f5f7fa;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 15px;
        }
        
        /* Navigation highlight */
        .nav-highlight {
            background-color: #4b8bf5;
            color: white !important;
            border-radius: 4px;
            padding: 8px 12px;
        }
        .nav-highlight:hover {
            background-color: #3a7ae0;
        }
        
        /* Button styling */
        .primary-button {
            background-color: #4b8bf5 !important;
            color: white !important;
        }
        
        .primary-button:hover {
            background-color: #3a7ae0 !important;
        }

        /* Hide file uploader label and make the dropzone minimal */
        [data-testid="stFileUploader"] {
            width: auto !important;
        }

        /* Hide the unnecessary text */
        [data-testid="stFileUploader"] section > div {
            display: none;
        }

        /* Style just the icon/button area */
        [data-testid="stFileUploader"] section {
            padding: 0 !important;
            border: none !important;
            display: flex;
            justify-content: flex-end;
        }

        /* Style for the "Process" button to be compact */
        .process-btn {
            padding: 0 8px !important;
            height: 36px !important;
            margin-top: 2px !important;
        }

        /* Align text input and file uploader vertically */
        .input-row {
            display: flex;
            align-items: center;
        }

        /* Make sure the file uploader doesn't take too much space */
        .file-upload-col {
            width: auto !important;
            flex-shrink: 0 !important;
        }

        /* Style for the custom paperclip icon */
        .paperclip-icon {
            cursor: pointer;
            margin-top: 6px;
            margin-left: 5px;
            font-size: 24px;
            color: #666;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # API endpoints
    API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
    LLM_API_URL = f"{API_BASE_URL}/llm"

    # Initialize session states
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = generate_unique_id()  # Use our custom function instead of uuid
        
    if 'parsed_documents' not in st.session_state:
        st.session_state.parsed_documents = {}
        
    if 'active_document' not in st.session_state:
        st.session_state.active_document = {}
    
    # Main layout
    st.title("🤖 Chat with AI")
    st.subheader("Ask questions about your documents")
    
    # Sidebar content specific to chat AI
    with st.sidebar:
        st.header("📊 Agent Settings")
        
        # LLM Model Selection
        llm_options = [(model["id"], f"{model['name']} ({model['provider']})") 
                        for model in st.session_state.available_llms]
        selected_model_option = st.selectbox(
            "Select LLM Model",
            options=[option[1] for option in llm_options],
            index=0
        )
        
        # Get the model ID from the selected option
        selected_model_index = [option[1] for option in llm_options].index(selected_model_option)
        llm_model_id = llm_options[selected_model_index][0]
        
        # Display token usage and pricing info
        st.info(f"Model usage is tracked and billed per token. Input tokens and output tokens have different pricing.")
        
        # Get model pricing from backend (in a real app)
        # For this example, we'll use hardcoded prices
        model_prices = {
            "gpt-4o": {"input": "$0.01/1K tokens", "output": "$0.03/1K tokens"},
            "claude-3-5-sonnet": {"input": "$0.008/1K tokens", "output": "$0.024/1K tokens"},
            "gpt-3.5-turbo": {"input": "$0.0005/1K tokens", "output": "$0.0015/1K tokens"},
            "llama-3-70b": {"input": "$0.0007/1K tokens", "output": "$0.0009/1K tokens"}
        }
        
        # Show pricing for selected model
        if llm_model_id in model_prices:
            price_info = model_prices[llm_model_id]
            st.markdown(f"""
                **Pricing for selected model:**
                - Input: {price_info['input']}
                - Output: {price_info['output']}
            """)
        
        # Track total token usage for the session
        if 'total_token_usage' not in st.session_state:
            st.session_state.total_token_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost": 0.0
            }
            
        # Display current session usage
        st.markdown("**Current Session Usage:**")
        st.markdown(f"""
            - Input tokens: {st.session_state.total_token_usage['input_tokens']}
            - Output tokens: {st.session_state.total_token_usage['output_tokens']}
            - Est. cost: ${st.session_state.total_token_usage['estimated_cost']:.4f}
        """)
        
    
        st.markdown("---")
        
        st.header("📁 Document Manager")
        
        # Document Selection
        if st.session_state.parsed_documents:
            st.subheader("Select and Download PDFs")
            
            doc_names = list(st.session_state.parsed_documents.keys())
            selected_doc = st.selectbox("Select a PDF document:", doc_names)
            
            if selected_doc:
                doc_info = st.session_state.parsed_documents[selected_doc]
                st.text(f"Size: {len(doc_info['content']) // 1000} KB")
                
                if st.button("Set as Active Document", use_container_width=True):
                    st.session_state.active_document = {
                        "name": selected_doc,
                        "content": doc_info["content"],
                        "type": doc_info["type"]
                    }
                    st.rerun()
        else:
            st.info("No documents parsed yet. Go to Data Parsing to extract content first.")
        
        # Conversation Controls
        st.markdown("---")
        st.header("💬 Conversation")
        
        if st.button("New Chat", use_container_width=True):
            new_conversation()
            st.rerun()
    
    # Main content area
    # Document Information
    if st.session_state.active_document:
        st.markdown(
            f"""
            <div class="document-selector">
                <p><strong>Current Document:</strong> {st.session_state.active_document["name"]} 
                ({st.session_state.active_document["type"].upper()})</p>
                <p>Content Length: {len(st.session_state.active_document["content"]) // 1000} KB</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Get Summary", type="primary", use_container_width=True):
                summarize_document(st.session_state.active_document["content"], llm_model_id)
        
        with col2:
            if st.button("Extract Key Points", type="primary", use_container_width=True):
                extract_key_points(st.session_state.active_document["content"], llm_model_id)
        
        with col3:
            if st.button("Generate Infographic", type="primary", use_container_width=True):
                st.info("Infographic generation will be implemented in a future update")
    else:
        st.warning("No document is currently active. Please select a document from the sidebar or parse a new document.")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Go to Data Parsing", type="primary", use_container_width=True):
                st.session_state.current_page = "data_parsing"
                st.rerun()
        
        with col2:
            if st.button("Get Summary", type="primary", use_container_width=True):
                st.info("Please select a document first to generate a summary.")
        
        with col3:
            if st.button("Extract Key Points", type="primary", use_container_width=True):
                st.info("Please select a document first to extract key points.")
    
    # Chat interface
    st.markdown("### Chat History")
    
    # Chat history with usage information
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.chat_history:
            # Add a download button for the conversation history with usage data
            if st.button("📥 Download Conversation with Usage Data", key="download_history"):
                # Create a formatted version of the chat history with usage data
                import json
                import base64
                from datetime import datetime
                
                # Format the chat history for download
                download_data = {
                    "conversation_id": st.session_state.conversation_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "active_document": st.session_state.active_document.get("name", "None"),
                    "messages": st.session_state.chat_history,
                    "token_usage": {
                        "total_input_tokens": st.session_state.total_token_usage["input_tokens"],
                        "total_output_tokens": st.session_state.total_token_usage["output_tokens"],
                        "total_cost": st.session_state.total_token_usage["estimated_cost"]
                    }
                }
                
                # Convert to JSON string
                json_str = json.dumps(download_data, indent=2)
                
                # Create download link
                b64 = base64.b64encode(json_str.encode()).decode()
                filename = f"conversation_{st.session_state.conversation_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                href = f'<a href="data:application/json;base64,{b64}" download="{filename}">Download JSON</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        # Display usage summary if there are messages
        if 'query_usage_history' in st.session_state and st.session_state.query_usage_history:
            with st.expander("📊 View Query Usage History", expanded=False):
                st.markdown("### Query Usage History")
                
                # Create a markdown table header
                usage_table = """
                | Query Type | Query | Model | Input Tokens | Output Tokens | Cost | Timestamp |
                | ---------- | ----- | ----- | ------------ | ------------- | ---- | --------- |
                """
                
                # Add each query's usage data to the table
                for usage in st.session_state.query_usage_history:
                    usage_table += f"| {usage['query_type']} | {usage['query']} | {usage['model']} | {usage['input_tokens']:,} | {usage['output_tokens']:,} | ${usage['cost']:.4f} | {usage['timestamp']} |\n"
                
                st.markdown(usage_table)
                
                # Calculate and display totals
                total_input = sum(usage['input_tokens'] for usage in st.session_state.query_usage_history)
                total_output = sum(usage['output_tokens'] for usage in st.session_state.query_usage_history)
                total_cost = sum(usage['cost'] for usage in st.session_state.query_usage_history)
                
                st.markdown(f"""
                **Summary:**
                - Total Input Tokens: {total_input:,}
                - Total Output Tokens: {total_output:,}
                - Total Cost: ${total_cost:.4f}
                """)
        
        # Display the chat messages
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f"""
                    <div class="user-message">
                        <p>{message["content"]}</p>
                        <div class="message-metadata">{message["timestamp"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif message["role"] == "assistant":
                # Get model display name for the message
                model_id = message.get("model", "unknown")
                model_display_name = "Unknown Model"
                
                for model in st.session_state.available_llms:
                    if model["id"] == model_id:
                        model_display_name = f"{model['name']} ({model['provider']})"
                        break
                
                # Display the message with enhanced metadata
                st.markdown(
                    f"""
                    <div class="assistant-message">
                        <p>{message["content"]}</p>
                        <div class="message-metadata">
                            {message.get("timestamp", "")} • 
                            Model: {model_display_name} • 
                            Tokens: {message.get("usage", {}).get("input_tokens", 0)} in / 
                            {message.get("usage", {}).get("output_tokens", 0)} out • 
                            Cost: ${message.get("usage", {}).get("cost", 0):.4f} • 
                            Time: {message.get("processing_time", 0):.1f}s
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif message["role"] == "system":
                st.markdown(
                    f"""
                    <div class="system-message">
                        {message["content"]} • {message["timestamp"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    # User input and file upload section
    st.markdown("### Ask a Question or Upload a New Document")

    # Create a container for the input field and upload button
    input_container = st.container()

    with input_container:
        # Use custom HTML for the layout
        st.markdown("""
        <div class="input-row">
            <div style="flex-grow: 1; margin-right: 10px;">
                <!-- Leave space for Streamlit to inject the input box here -->
            </div>
            <div class="file-upload-col">
                <!-- Leave space for Streamlit to inject the file uploader here -->
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Define process question function
        def process_question():
            """Process the user question when Enter is pressed"""
            user_question = st.session_state.user_question_input
            
            if user_question:
                if not st.session_state.active_document:
                    st.error("Please select a document first.")
                    return
                    
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_question,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                })
                
                # Call LLM API
                with st.spinner("Thinking..."):
                    result = query_llm(
                        prompt=user_question,
                        model=llm_model_id,
                        document_content=st.session_state.active_document["content"],
                        operation_type="chat"
                    )
                    
                    if result:
                        # Add assistant response to chat history
                        message_with_tokens = {
                            "role": "assistant",
                            "content": result["text"],
                            "usage": {
                                "input_tokens": result["input_tokens"],
                                "output_tokens": result["output_tokens"],
                                "cost": result["cost"]
                            },
                            "model": result["model"],
                            "processing_time": result["processing_time"],
                            "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                        }
                        st.session_state.chat_history.append(message_with_tokens)
                        
                        # Add usage information for this query to the session state
                        if 'query_usage_history' not in st.session_state:
                            st.session_state.query_usage_history = []
                        
                        st.session_state.query_usage_history.append({
                            "query_type": "chat",
                            "query": user_question[:50] + "..." if len(user_question) > 50 else user_question,
                            "model": result["model"],
                            "input_tokens": result["input_tokens"],
                            "output_tokens": result["output_tokens"],
                            "cost": result["cost"],
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        # Clear the input field
                        st.session_state.user_question_input = ""
        
        # Create a layout with columns for input and file upload
        col1, col2 = st.columns([20, 1])
        
        with col1:
            # Use on_change event for text input
            user_input = st.text_input(
                "Type your question about the document and press Enter",
                placeholder="What is the main topic of this document?",
                key="user_question_input",
                on_change=process_question,
                label_visibility="collapsed"
            )
        
        with col2:
            # Add minimal file uploader with just the icon visible
            uploaded_file = st.file_uploader("", type=["pdf"], key="quick_upload", label_visibility="collapsed")
            
            # # Display paperclip icon using a Unicode paperclip: 📎
            # st.markdown('<div class="paperclip-icon">📎</div>', unsafe_allow_html=True)
            
            if uploaded_file:
                # Show a process button only if a file is uploaded
                st.markdown("<div style='height: 5px'></div>", unsafe_allow_html=True)
                if st.button("Process", key="quick_process", use_container_width=True):
                    with st.spinner("Processing..."):
                        import time
                        time.sleep(2)
                        
                        # Create a unique document ID
                        doc_id = generate_unique_id()
                        
                        # Simulated extraction content
                        extracted_content = f"Content from {uploaded_file.name}. This is a placeholder."
                        
                        # Add to parsed documents
                        st.session_state.parsed_documents[uploaded_file.name] = {
                            "content": extracted_content,
                            "type": "pdf",
                            "extraction_engine": "quick_upload",
                            "date_added": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "id": doc_id
                        }
                        
                        # Set as active document
                        st.session_state.active_document = {
                            "name": uploaded_file.name,
                            "content": extracted_content,
                            "type": "pdf",
                            "id": doc_id
                        }
                        
                        st.success(f"Document processed")
                        st.rerun()

    
    # Display token usage and cost information
    with st.expander("💰 View Token Usage and Pricing Details", expanded=False):
        st.markdown("### Token Usage and Pricing Information")
        st.markdown("""
        LLMs charge based on the number of tokens processed. Tokens are pieces of words, and pricing varies by model:
        
        - **Input tokens**: Text sent to the model (your question + document context)
        - **Output tokens**: Text generated by the model (the response)
        """)
        
        # Create a table showing pricing for available models
        st.markdown("#### Model Pricing (per 1,000 tokens)")
        
        pricing_data = []
        for model in st.session_state.available_llms:
            model_id = model["id"]
            # This would come from your actual pricing data in a real implementation
            input_price = "$0.01" if model_id == "gpt-4o" else "$0.008" if model_id == "claude-3-5-sonnet" else "$0.0005"
            output_price = "$0.03" if model_id == "gpt-4o" else "$0.024" if model_id == "claude-3-5-sonnet" else "$0.0015"
            
            pricing_data.append(f"| {model['name']} | {input_price} | {output_price} |")
        
        pricing_table = """
        | Model | Input Price | Output Price |
        | ----- | ----------- | ------------ |
        """ + "\n".join(pricing_data)
        
        st.markdown(pricing_table)
        
        # Show current session usage
        st.markdown("#### Current Session Usage")
        st.markdown(f"""
        - **Total Input Tokens**: {st.session_state.total_token_usage['input_tokens']:,}
        - **Total Output Tokens**: {st.session_state.total_token_usage['output_tokens']:,}
        - **Estimated Total Cost**: ${st.session_state.total_token_usage['estimated_cost']:.4f}
        """)

def query_llm(prompt, model="gpt-4o", document_content=None, operation_type="chat"):
    """
    Send a query to the LLM API
    Parameters:
    - prompt: The user's question or instruction
    - model: LLM model to use
    - document_content: Document content for context
    - operation_type: Type of operation (chat, summarize, extract_key_points)
    """
    API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
    API_BACKEND_URL = os.getenv('API_BACKEND_URL', 'http://localhost:8000')
    LLM_API_URL = f"{API_BACKEND_URL}/llm"
    
    # Select the appropriate endpoint based on operation type
    if operation_type == "chat":
        endpoint = f"{LLM_API_URL}/ask_question"
    elif operation_type == "summarize":
        endpoint = f"{LLM_API_URL}/summarize"
    elif operation_type == "extract_key_points":
        endpoint = f"{LLM_API_URL}/extract_keypoints"
    else:
        endpoint = f"{LLM_API_URL}/generate"
    
    try:
        # Prepare the payload
        payload = {
            "prompt": prompt,
            "model": model,
            "operation_type": operation_type,
            "conversation_id": st.session_state.conversation_id
        }
        
        # Include document content if available
        if document_content:
            payload["document_content"] = document_content
            
        # If we have an active document with an ID, include it
        if 'active_document' in st.session_state and st.session_state.active_document.get('id'):
            payload["document_id"] = st.session_state.active_document.get('id')
            
        # Make the API request
        response = requests.post(
            endpoint,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract token usage information
            input_tokens = result.get("usage", {}).get("input_tokens", 0)
            output_tokens = result.get("usage", {}).get("output_tokens", 0)
            cost = result.get("usage", {}).get("cost", 0)
            
            # Update total token usage for the session
            st.session_state.total_token_usage["input_tokens"] += input_tokens
            st.session_state.total_token_usage["output_tokens"] += output_tokens
            st.session_state.total_token_usage["estimated_cost"] += cost
            
            # Log token usage for this request
            print(f"LLM Request - Model: {model}, Input tokens: {input_tokens}, Output tokens: {output_tokens}, Cost: ${cost:.6f}")
            
            return {
                "text": result.get("response", ""),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "processing_time": result.get("processing_time", 0),
                "model": model
            }
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error communicating with LLM API: {str(e)}")
        return None

def new_conversation():
    """Create a new conversation"""
    st.session_state.chat_history = []
    st.session_state.conversation_id = generate_unique_id()  # Use our custom ID generator

def summarize_document(document_content, model):
    """Generate a summary of the document"""
    with st.spinner("Generating document summary..."):
        result = query_llm(
            prompt="Provide a comprehensive summary of this document.",
            model=model,
            document_content=document_content,
            operation_type="summarize"
        )
        
        if result:
            # Add to chat history with detailed token usage information
            message_with_tokens = {
                "role": "system",
                "content": "Document summary generated",
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            }
            st.session_state.chat_history.append(message_with_tokens)
            
            message_with_tokens = {
                "role": "assistant",
                "content": result["text"],
                "usage": {
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "cost": result["cost"]
                },
                "model": result["model"],
                "processing_time": result["processing_time"],
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            }
            st.session_state.chat_history.append(message_with_tokens)
            
            # Add usage information for this query to the session state
            if 'query_usage_history' not in st.session_state:
                st.session_state.query_usage_history = []
            
            st.session_state.query_usage_history.append({
                "query_type": "summarize",
                "query": "Document summary request",
                "model": result["model"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "cost": result["cost"],
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            return result
        return None

def extract_key_points(document_content, model):
    """Extract key points from the document"""
    with st.spinner("Extracting key points..."):
        result = query_llm(
            prompt="Extract the key points from this document.",
            model=model,
            document_content=document_content,
            operation_type="extract_key_points"
        )
        
        if result:
            # Add to chat history with detailed token usage information
            message_with_tokens = {
                "role": "system",
                "content": "Key points extracted",
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            }
            st.session_state.chat_history.append(message_with_tokens)
            
            message_with_tokens = {"role": "assistant",
                "content": result["text"],
                "usage": {
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "cost": result["cost"]
                },
                "model": result["model"],
                "processing_time": result["processing_time"],
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            }
            st.session_state.chat_history.append(message_with_tokens)
            
            # Add usage information for this query to the session state
            if 'query_usage_history' not in st.session_state:
                st.session_state.query_usage_history = []
            
            st.session_state.query_usage_history.append({
                "query_type": "keypoints",
                "query": "Key points extraction request",
                "model": result["model"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "cost": result["cost"],
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            return result
        return None























