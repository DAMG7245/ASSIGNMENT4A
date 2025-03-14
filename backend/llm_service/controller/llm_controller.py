import os
import uuid
import time
import json
from typing import Dict, Any, List, Generator, Optional
from fastapi import BackgroundTasks
import logging
import litellm
from litellm import completion, acompletion

# Configure logging
logger = logging.getLogger(__name__)

# Configure LiteLLM
litellm.set_verbose = True

class LLMController:
    """Controller for LLM interactions using LiteLLM"""
    
    def __init__(self):
        """Initialize the LiteLLM configuration and task storage"""
        # Load API keys from environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.azure_api_key = os.getenv("AZURE_API_KEY")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        
        # Configure LiteLLM
        self._configure_litellm()
        
        # Default model for each provider
        self.default_models = {
            "openai": "gpt-4",
            "anthropic": "claude-3-opus-20240229",
            "google": "gemini-pro",
            "azure": "azure/gpt-4",
            "cohere": "command",
        }
        
        # In-memory task storage
        self.tasks = {}
        
        logger.info("LLMController initialized successfully")
    
    def _configure_litellm(self):
        """Configure LiteLLM with API keys"""
        # Set API keys
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        
        if self.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key
        
        if self.google_api_key:
            os.environ["GOOGLE_API_KEY"] = self.google_api_key
        
        if self.azure_api_key:
            os.environ["AZURE_API_KEY"] = self.azure_api_key
        
        if self.cohere_api_key:
            os.environ["COHERE_API_KEY"] = self.cohere_api_key
        
        # Set default configuration
        litellm.drop_params = True  # Drop unsupported params instead of error
        litellm.num_retries = 3      # Retry API calls 3 times
        
        # Optional: Load from config file if exists
        config_path = os.getenv("LITELLM_CONFIG_PATH")
        if config_path and os.path.exists(config_path):
            try:
                litellm.config_path = config_path
                logger.info(f"Loaded LiteLLM config from {config_path}")
            except Exception as e:
                logger.error(f"Error loading LiteLLM config: {str(e)}")
    
    def generate(self, prompt: str, model: str = None, max_tokens: int = 1000, 
                temperature: float = 0.7, provider: str = None, 
                additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate text using LiteLLM
        
        Args:
            prompt: The input prompt
            model: Model name
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter for generation
            provider: Override provider detection
            additional_params: Additional model-specific parameters
            
        Returns:
            Dict containing generated text and metadata
        """
        # Determine model to use
        model_name = self._get_model_name(model, provider)
        
        # Prepare parameters
        params = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Add additional parameters
        if additional_params:
            params.update(additional_params)
        
        try:
            logger.info(f"Generating text with model: {model_name}")
            response = completion(**params)
            
            # Get model provider
            provider = self._extract_provider(response, model_name)
            
            # Extract and format response
            result = {
                "text": response.choices[0].message.content,
                "model": response.model,
                "provider": provider,
                "finish_reason": response.choices[0].finish_reason
            }
            
            # Add usage information if available
            if hasattr(response, "usage") and response.usage:
                result["usage"] = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0)
                }
            
            logger.info(f"Successfully generated text with provider: {provider}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise Exception(f"Error generating text: {str(e)}")
    
    def chat(self, messages: List[Dict[str, str]], model: str = None, max_tokens: int = 1000, 
            temperature: float = 0.7, provider: str = None, 
            additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate chat completions
        
        Args:
            messages: List of message objects with role and content
            model: Model name
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter for generation
            provider: Override provider detection
            additional_params: Additional model-specific parameters
            
        Returns:
            Dict containing generated text and metadata
        """
        # Determine model to use
        model_name = self._get_model_name(model, provider)
        
        # Prepare parameters
        params = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Add additional parameters
        if additional_params:
            params.update(additional_params)
        
        try:
            logger.info(f"Generating chat completion with model: {model_name}")
            response = completion(**params)
            
            # Get model provider
            provider = self._extract_provider(response, model_name)
            
            # Extract and format response
            result = {
                "text": response.choices[0].message.content,
                "model": response.model,
                "provider": provider,
                "finish_reason": response.choices[0].finish_reason
            }
            
            # Add usage information if available
            if hasattr(response, "usage") and response.usage:
                result["usage"] = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0)
                }
            
            logger.info(f"Successfully generated chat completion with provider: {provider}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating chat completion: {str(e)}")
            raise Exception(f"Error generating chat completion: {str(e)}")
    
    def stream(self, messages: List[Dict[str, str]], model: str = None, max_tokens: int = 1000, 
              temperature: float = 0.7, provider: str = None, 
              additional_params: Dict[str, Any] = None) -> Generator[str, None, None]:
        """
        Stream responses from LLM models
        
        Args:
            messages: List of message objects with role and content
            model: Model name
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter for generation
            provider: Override provider detection
            additional_params: Additional model-specific parameters
            
        Yields:
            JSON strings with chunks of the response
        """
        # Determine model to use
        model_name = self._get_model_name(model, provider)
        
        # Prepare parameters
        params = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        
        # Add additional parameters
        if additional_params:
            params.update(additional_params)
        
        try:
            logger.info(f"Streaming response from model: {model_name}")
            
            # Start streaming
            for chunk in completion(**params):
                # Extract content from the chunk
                content = ""
                if chunk.choices and len(chunk.choices) > 0:
                    if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
                        content = chunk.choices[0].delta.content or ""
                
                # Create streamable JSON
                stream_data = {
                    "content": content,
                    "model": model_name,
                    "provider": self._extract_provider(None, model_name)
                }
                
                yield json.dumps(stream_data)
                
        except Exception as e:
            logger.error(f"Error streaming response: {str(e)}")
            yield json.dumps({"error": str(e)})
    
    def list_models(self) -> List[Dict[str, str]]:
        """
        List all available models
        
        Returns:
            List of model objects with id, provider, and description
        """
        try:
            # Get models from LiteLLM
            models_info = []
            
            # Add OpenAI models
            if self.openai_api_key:
                models_info.extend([
                    {"id": "gpt-4", "provider": "openai", "description": "OpenAI GPT-4"},
                    {"id": "gpt-4-turbo", "provider": "openai", "description": "OpenAI GPT-4 Turbo"},
                    {"id": "gpt-3.5-turbo", "provider": "openai", "description": "OpenAI GPT-3.5 Turbo"}
                ])
            
            # Add Anthropic models
            if self.anthropic_api_key:
                models_info.extend([
                    {"id": "claude-3-opus-20240229", "provider": "anthropic", "description": "Anthropic Claude 3 Opus"},
                    {"id": "claude-3-sonnet-20240229", "provider": "anthropic", "description": "Anthropic Claude 3 Sonnet"},
                    {"id": "claude-3-haiku-20240307", "provider": "anthropic", "description": "Anthropic Claude 3 Haiku"}
                ])
            
            # Add Google models
            if self.google_api_key:
                models_info.extend([
                    {"id": "gemini-pro", "provider": "google", "description": "Google Gemini Pro"},
                    {"id": "gemini-ultra", "provider": "google", "description": "Google Gemini Ultra"}
                ])
            
            # Add Cohere models
            if self.cohere_api_key:
                models_info.extend([
                    {"id": "command", "provider": "cohere", "description": "Cohere Command"},
                    {"id": "command-r", "provider": "cohere", "description": "Cohere Command-R"},
                    {"id": "command-light", "provider": "cohere", "description": "Cohere Command Light"}
                ])
            
            return models_info
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            raise Exception(f"Error listing models: {str(e)}")
    
    def generate_async(self, background_tasks: BackgroundTasks, prompt: str, model: str = None, 
                      max_tokens: int = 1000, temperature: float = 0.7, provider: str = None, 
                      additional_params: Dict[str, Any] = None) -> str:
        """
        Generate text asynchronously
        
        Args:
            background_tasks: FastAPI BackgroundTasks
            prompt: The input prompt
            model: Model name
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter for generation
            provider: Override provider detection
            additional_params: Additional model-specific parameters
            
        Returns:
            Task ID for tracking the async task
        """
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task in storage
        self.tasks[task_id] = {
            "status": "processing",
            "created_at": time.time(),
            "result": None
        }
        
        # Add generation task to background tasks
        background_tasks.add_task(
            self._generate_async_task,
            task_id=task_id,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            provider=provider,
            additional_params=additional_params
        )
        
        return task_id
    
    async def _generate_async_task(self, task_id: str, prompt: str, model: str = None, 
                                max_tokens: int = 1000, temperature: float = 0.7, 
                                provider: str = None, additional_params: Dict[str, Any] = None):
        """Background task for async generation"""
        try:
            # Run generation
            result = self.generate(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                provider=provider,
                additional_params=additional_params
            )
            
            # Update task with result
            self.tasks[task_id] = {
                "status": "completed",
                "created_at": self.tasks[task_id]["created_at"],
                "completed_at": time.time(),
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Async generation error: {str(e)}")
            
            # Update task with error
            self.tasks[task_id] = {
                "status": "failed",
                "created_at": self.tasks[task_id]["created_at"],
                "completed_at": time.time(),
                "error": str(e)
            }
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of an async task
        
        Args:
            task_id: The task ID
            
        Returns:
            Dict containing task status and result (if available)
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task ID {task_id} not found")
        
        return self.tasks[task_id]
    
    def _get_model_name(self, model: Optional[str], provider: Optional[str]) -> str:
        """
        Determine the model name to use
        
        Args:
            model: Model name (optional)
            provider: Provider name (optional)
            
        Returns:
            Model name to use with LiteLLM
        """
        if model:
            return model
        
        if provider and provider in self.default_models:
            return self.default_models[provider]
        
        # Use OpenAI as default if available
        if self.openai_api_key:
            return self.default_models["openai"]
        
        # Otherwise, use the first available provider
        for provider, model in self.default_models.items():
            env_var = f"{provider.upper()}_API_KEY"
            if os.getenv(env_var):
                return model
        
        # If no API keys are available
        raise ValueError("No LLM API keys configured. Please set at least one provider API key.")
    
    def _extract_provider(self, response: Any, model_name: str) -> str:
        """
        Extract provider name from response or model name
        
        Args:
            response: LiteLLM response object
            model_name: Model name used
            
        Returns:
            Provider name
        """
        # Try to extract from response
        if response and hasattr(response, "model"):
            model = response.model.lower()
            
            if "gpt" in model or "openai" in model:
                return "openai"
            elif "claude" in model or "anthropic" in model:
                return "anthropic"
            elif "gemini" in model or "google" in model:
                return "google"
            elif "command" in model or "cohere" in model:
                return "cohere"
            elif "azure" in model:
                return "azure"
        
        # Extract from model name
        model = model_name.lower()
        
        if "gpt" in model or model.startswith("openai/"):
            return "openai"
        elif "claude" in model or model.startswith("anthropic/"):
            return "anthropic"
        elif "gemini" in model or model.startswith("google/"):
            return "google"
        elif "command" in model or model.startswith("cohere/"):
            return "cohere"
        elif "azure" in model or model.startswith("azure/"):
            return "azure"
        
        # Default fallback
        return "unknown"