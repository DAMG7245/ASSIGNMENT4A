import os
import uuid
import time
import json
from typing import Dict, Any, List, Generator, Optional
from fastapi import BackgroundTasks
import logging
import litellm
from litellm import completion, acompletion
from dotenv import load_dotenv

load_dotenv()
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
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.xai_api_key = os.getenv("XAI_API_KEY")  # For Grok models
        
        print(f"Google API Key loaded: {'[PRESENT]' if self.gemini_api_key else '[MISSING]'}")
    # For debugging only, don't print the full key in production
        if self.gemini_api_key:
            print(f"First few chars of key: {self.gemini_api_key}")
        # Configure LiteLLM
        self._configure_litellm()
        
        # Default model for each provider - updated with correct working formats
        self.default_models = {
            "openai": "gpt-4o",                # OpenAI works without prefix
            "anthropic": "claude-3-5-sonnet-20240620",  # Anthropic works without prefix, but model needs to exist
            "gemini": "gemini-1.5-pro",            # Need to install gemini-generativeai
            "deepseek": "deepseek-coder",      # Need deepseek integration
            "xai": "grok-1"                    # Need xAI integration
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
        
        if self.gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = self.gemini_api_key
        
        if self.deepseek_api_key:
            os.environ["DEEPSEEK_API_KEY"] = self.deepseek_api_key
            
        if self.xai_api_key:
            os.environ["XAI_API_KEY"] = self.xai_api_key
        
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
        # Extract the model name and provider
        model_name, provider_name = self._parse_model_and_provider(model, provider)
        
        # Add a fallback for when API keys are missing
        # Check if this is an OpenAI model but we don't have an API key
        if provider_name == "openai" and not self.openai_api_key:
            logger.warning(f"OpenAI API key not set. Using mock response for model: {model_name}")
            return self._get_mock_response(model_name, provider_name)
        
        # Similar checks for other providers
        if provider_name == "anthropic" and not self.anthropic_api_key:
            logger.warning(f"Anthropic API key not set. Using mock response for model: {model_name}")
            return self._get_mock_response(model_name, provider_name)
        
        if provider_name == "gemini" and not self.gemini_api_key:
            logger.warning(f"Google API key not set. Using mock response for model: {model_name}")
            return self._get_mock_response(model_name, provider_name)
            
        if provider_name == "deepseek" and not self.deepseek_api_key:
            logger.warning(f"DeepSeek API key not set. Using mock response for model: {model_name}")
            return self._get_mock_response(model_name, provider_name)
            
        if provider_name == "xai" and not self.xai_api_key:
            logger.warning(f"xAI API key not set. Using mock response for model: {model_name}")
            return self._get_mock_response(model_name, provider_name)
        
        # Initialize additional_params if None
        if additional_params is None:
            additional_params = {}
        
        # Prepare parameters
        params = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Only set custom_llm_provider if it's not OpenAI (default in LiteLLM)
        if provider_name and provider_name.lower() != "openai":
            params["custom_llm_provider"] = provider_name.lower()
        
        # IMPORTANT FIX: For Gemini, explicitly pass the API key in the parameters
        if provider_name.lower() == "gemini" and self.gemini_api_key:
            params["api_key"] = self.gemini_api_key
            
            # For Gemini provider, we need to be explicit about the format
            # Set the model name exactly as expected by the API
            params["model"] = "gemini-1.5-pro"
            
            # Make sure the LiteLLM knows to use Gemini provider
            params["custom_llm_provider"] = "gemini"
        if provider_name.lower() == "anthropic":
            params["api_type"] = "anthropic_messages"
            params["api_base"] = "https://api.anthropic.com/v1"

        # Add additional parameters
        if additional_params:
            params.update(additional_params)
        
        try:
            logger.info(f"Generating text with model: {model_name} and provider: {provider_name}")
            
            # Log actual parameters being sent to litellm (without showing the full API key)
            safe_params = params.copy()
            if "api_key" in safe_params:
                safe_params["api_key"] = "[REDACTED]"
            logger.debug(f"LiteLLM parameters: {safe_params}")
            
            # For Gemini specifically, print some extra debugging info
            if provider_name.lower() == "gemini":
                logger.info(f"Using Gemini API with key present: {bool(self.gemini_api_key)}")
                logger.info(f"Gemini model name being used: {params['model']}")
            
            response = completion(**params)
            
            # Extract and format response
            result = {
                "text": response.choices[0].message.content,
                "model": response.model,
                "provider": provider_name,
                "finish_reason": response.choices[0].finish_reason
            }
            
            # Add usage information if available
            if hasattr(response, "usage") and response.usage:
                result["usage"] = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0)
                }
            
            logger.info(f"Successfully generated text with provider: {provider_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            
            # Provide a fallback response in case of any error
            return {
                "text": f"Error generating response: {str(e)}. This is a fallback response.",
                "model": model_name,
                "provider": "error_fallback",
                "finish_reason": "error",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
        
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
        # Extract the model name and provider
        model_name, provider_name = self._parse_model_and_provider(model, provider)
        
        # Add a fallback for when API keys are missing
        if provider_name == "openai" and not self.openai_api_key:
            logger.warning(f"OpenAI API key not set. Using mock response for model: {model_name}")
            return self._get_mock_response(model_name, provider_name)
        
        # Similar checks for other providers
        if provider_name == "anthropic" and not self.anthropic_api_key:
            logger.warning(f"Anthropic API key not set. Using mock response for model: {model_name}")
            return self._get_mock_response(model_name, provider_name)
            
        if provider_name == "gemini" and not self.gemini_api_key:
            logger.warning(f"Google API key not set. Using mock response for model: {model_name}")
            return self._get_mock_response(model_name, provider_name)
            
        if provider_name == "deepseek" and not self.deepseek_api_key:
            logger.warning(f"DeepSeek API key not set. Using mock response for model: {model_name}")
            return self._get_mock_response(model_name, provider_name)
            
        if provider_name == "xai" and not self.xai_api_key:
            logger.warning(f"xAI API key not set. Using mock response for model: {model_name}")
            return self._get_mock_response(model_name, provider_name)
        
        # Initialize additional_params if None
        if additional_params is None:
            additional_params = {}
        
        # Prepare parameters
        params = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Only set custom_llm_provider if it's not OpenAI (default in LiteLLM)
        if provider_name and provider_name.lower() != "openai":
            params["custom_llm_provider"] = provider_name.lower()
        
        # IMPORTANT FIX: For Gemini, explicitly pass the API key in the parameters
        if provider_name.lower() == "gemini" and self.gemini_api_key:
            params["api_key"] = self.gemini_api_key
            
            # For Gemini provider, we need to be explicit about the format
            # Set the model name exactly as expected by the API
            params["model"] = "gemini-1.5-pro"
            
            # Make sure the LiteLLM knows to use Gemini provider
            params["custom_llm_provider"] = "gemini"
        if provider_name.lower() == "anthropic":
            params["api_type"] = "anthropic_messages"
            params["api_base"] = "https://api.anthropic.com/v1"

        # Add additional parameters
        if additional_params:
            params.update(additional_params)
        
        try:
            logger.info(f"Generating chat completion with model: {model_name} and provider: {provider_name}")
            
            # Log actual parameters being sent to litellm (without showing the full API key)
            safe_params = params.copy()
            if "api_key" in safe_params:
                safe_params["api_key"] = "[REDACTED]"
            logger.debug(f"LiteLLM parameters: {safe_params}")
            
            # For Gemini specifically, print some extra debugging info
            if provider_name.lower() == "gemini":
                logger.info(f"Using Gemini API with key present: {bool(self.gemini_api_key)}")
                logger.info(f"Gemini model name being used: {params['model']}")
            
            response = completion(**params)
            
            # Extract and format response
            result = {
                "text": response.choices[0].message.content,
                "model": response.model,
                "provider": provider_name,
                "finish_reason": response.choices[0].finish_reason
            }
            
            # Add usage information if available
            if hasattr(response, "usage") and response.usage:
                result["usage"] = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0)
                }
            
            logger.info(f"Successfully generated chat completion with provider: {provider_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating chat completion: {str(e)}")
            
            # Provide a fallback response in case of any error
            return {
                "text": f"Error generating response: {str(e)}. This is a fallback response.",
                "model": model_name,
                "provider": "error_fallback",
                "finish_reason": "error",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
    def _get_mock_response(self, model_name, provider_name):
        """Generate a mock response when API keys are missing"""
        return {
            "text": "This is a mock response generated because no valid API key was found. In a real environment, this would be a response based on the document content.",
            "model": model_name,
            "provider": "mock",
            "finish_reason": "mock_complete",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
    
    def _parse_model_and_provider(self, model: Optional[str], provider: Optional[str]) -> tuple:
        """
        Parse the model name and provider, ensuring both are properly set
        
        Args:
            model: Model name (optional)
            provider: Provider name (optional)
            
        Returns:
            Tuple of (model_name, provider_name)
        """
        # If model contains a provider prefix (e.g., "anthropic/claude-3-5-sonnet-20240620")
        if model and "/" in model:
            provider_prefix, model_name = model.split("/", 1)
            return model_name, provider_prefix
        
        # If both model and provider are explicitly provided
        if model and provider:
            return model, provider
        
        # If only model is provided, try to infer the provider
        if model:
            model_provider = self._infer_provider_from_model(model)
            return model, model_provider
        
        # If only provider is provided, use the default model for that provider
        if provider and provider.lower() in self.default_models:
            return self.default_models[provider.lower()], provider.lower()
        
        # Default behavior: use OpenAI if available, otherwise pick the first available provider
        if self.openai_api_key:
            return self.default_models["openai"], "openai"
        elif self.anthropic_api_key:
            return self.default_models["anthropic"], "anthropic"
        elif self.gemini_api_key:
            return self.default_models["gemini"], "gemini"
        elif self.deepseek_api_key:
            return self.default_models["deepseek"], "deepseek"
        elif self.xai_api_key:
            return self.default_models["xai"], "xai"
        
        # If no API keys are available, default to OpenAI for mock responses
        logger.warning("No LLM API keys configured. Using gpt-4o/openai for mock responses.")
        return self.default_models["openai"], "openai"
    def _infer_provider_from_model(self, model_name: str) -> str:
        """
        Infer the provider from the model name
        
        Args:
            model_name: Model name
            
        Returns:
            Provider name
        """
        model = model_name.lower()
        
        if "gpt" in model:
            return "openai"
        elif "claude" in model:
            return "anthropic"
        elif "gemini" in model:
            return "gemini"
        elif "deepseek" in model:
            return "deepseek"
        elif "grok" in model:
            return "xai"
        
        # Default fallback
        return "unknown"
    
    def list_models(self) -> List[Dict[str, str]]:
        """
        List all available models
        
        Returns:
            List of model objects with id, provider, and description
        """
        try:
            # Get models from LiteLLM
            models_info = []
            
            # Add only the 5 specific models with formats that work with litellm
            if self.openai_api_key:
                models_info.append(
                    {"id": "openai/gpt-4o", "provider": "openai", "description": "OpenAI GPT-4o"}
                )
            
            if self.anthropic_api_key:
                models_info.append(
                    {"id": "anthropic/claude-3-5-sonnet-20240620", "provider": "anthropic", "description": "Claude 3.5 Sonnet"}
                )
            
            if self.gemini_api_key:
                models_info.append(
                    {"id": "gemini/gemini-1.5-pro", "provider": "gemini", "description": "Google Gemini Pro"}
                )
                
            if self.deepseek_api_key:
                models_info.append(
                    {"id": "deepseek/deepseek-coder", "provider": "deepseek", "description": "DeepSeek Coder"}
                )
                
            if self.xai_api_key:
                models_info.append(
                    {"id": "xai/grok-1", "provider": "xai", "description": "xAI Grok"}
                )
            
            # Always include all 5 models for testing, even if API keys are missing
            if len(models_info) < 5:
                model_ids = [model["id"] for model in models_info]
                
                if "openai/gpt-4o" not in model_ids:
                    models_info.append({"id": "openai/gpt-4o", "provider": "openai", "description": "OpenAI GPT-4o (Mock)"})
                
                if "anthropic/claude-3-5-sonnet-20240620" not in model_ids:
                    models_info.append({"id": "anthropic/claude-3-5-sonnet-20240620", "provider": "anthropic", "description": "Claude 3.5 Sonnet (Mock)"})
                
                if "gemini/gemini-1.5-pro" not in model_ids:
                    models_info.append({"id": "gemini/gemini-1.5-pro", "provider": "gemini", "description": "Google Gemini Pro (Mock)"})
                    
                if "deepseek/deepseek-coder" not in model_ids:
                    models_info.append({"id": "deepseek/deepseek-coder", "provider": "deepseek", "description": "DeepSeek Coder (Mock)"})
                    
                if "xai/grok-1" not in model_ids:
                    models_info.append({"id": "xai/grok-1", "provider": "xai", "description": "xAI Grok (Mock)"})
            
            return models_info
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            # Return the 5 models in case of error - with correct formats
            return [
                {"id": "openai/gpt-4o", "provider": "openai", "description": "OpenAI GPT-4o (Fallback)"},
                {"id": "anthropic/claude-3-5-sonnet-20240620", "provider": "anthropic", "description": "Claude 3.5 Sonnet (Fallback)"},
                {"id": "gemini/gemini-1.5-pro", "provider": "gemini", "description": "Google Gemini Pro (Fallback)"},
                {"id": "deepseek/deepseek-coder", "provider": "deepseek", "description": "DeepSeek Coder (Fallback)"},
                {"id": "xai/grok-1", "provider": "xai", "description": "xAI Grok (Fallback)"}
            ]
    
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