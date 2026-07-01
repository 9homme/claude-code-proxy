import asyncio
import json
import time
from fastapi import HTTPException
from typing import Optional, AsyncGenerator, Dict, Any, Union, List
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai._exceptions import APIError, RateLimitError, AuthenticationError, BadRequestError
from src.core.logging import logger

class OpenAIClient:
    """Async OpenAI client with cancellation support."""
    
    def __init__(self, timeout: int = 90, api_version: Optional[str] = None, custom_headers: Optional[Dict[str, str]] = None):
        self.timeout = timeout
        self.api_version = api_version
        self.custom_headers = custom_headers or {}
        self.clients: Dict[str, Union[AsyncOpenAI, AsyncAzureOpenAI]] = {}
        self.active_requests: Dict[str, asyncio.Event] = {}
    
    def _get_client(self, api_key: str, base_url: str) -> Union[AsyncOpenAI, AsyncAzureOpenAI]:
        """Get or create an OpenAI client for the given credentials."""
        client_key = f"{api_key}:{base_url}"
        if client_key not in self.clients:
            # Prepare default headers
            default_headers = {
                "Content-Type": "application/json",
                "User-Agent": "claude-proxy/1.0.0"
            }
            
            # Merge custom headers with default headers
            all_headers = {**default_headers, **self.custom_headers}
            
            if self.api_version:
                self.clients[client_key] = AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=base_url,
                    api_version=self.api_version,
                    timeout=self.timeout,
                    default_headers=all_headers
                )
            else:
                self.clients[client_key] = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self.timeout,
                    default_headers=all_headers
                )
        return self.clients[client_key]

    async def create_chat_completion(self, request: Dict[str, Any], api_key: str, base_url: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Send chat completion to OpenAI API with cancellation support."""
        
        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event
        
        client = self._get_client(api_key, base_url)
        start_time = time.time()
        try:
            # Create task that can be cancelled
            completion_task = asyncio.create_task(
                client.chat.completions.create(**request)
            )
            
            if request_id:
                # Wait for either completion or cancellation
                cancel_task = asyncio.create_task(cancel_event.wait())
                done, pending = await asyncio.wait(
                    [completion_task, cancel_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Check if request was cancelled
                if cancel_task in done:
                    completion_task.cancel()
                    raise HTTPException(status_code=499, detail="Request cancelled by client")
                
                completion = await completion_task
            else:
                completion = await completion_task
            
            duration = time.time() - start_time
            logger.info(f"LLM request completed in {duration:.2f}s (model: {request.get('model')})")
            
            comp_dict = completion.model_dump()
            try:
                with open("last_openai_response.json", "w", encoding="utf-8") as f:
                    json.dump(comp_dict, f, indent=2, ensure_ascii=False)
            except Exception as e_log:
                logger.error(f"Failed to write last_openai_response.json: {e_log}")
            
            # Convert to dict format that matches the original interface
            return comp_dict
        
        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
        except RateLimitError as e:
            raise HTTPException(status_code=429, detail=self.classify_openai_error(str(e)))
        except BadRequestError as e:
            raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
        except APIError as e:
            status_code = getattr(e, 'status_code', 500)
            raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
        
        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def create_chat_completion_stream(self, request: Dict[str, Any], api_key: str, base_url: str, request_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Send streaming chat completion to OpenAI API with cancellation support."""
        
        # Create cancellation token if request_id provided
        if request_id:
            cancel_event = asyncio.Event()
            self.active_requests[request_id] = cancel_event
        
        client = self._get_client(api_key, base_url)
        start_time = time.time()
        ttft_logged = False
        try:
            # Ensure stream is enabled
            request["stream"] = True
            if "stream_options" not in request:
                request["stream_options"] = {}
            request["stream_options"]["include_usage"] = True
            
            # Create the streaming completion
            streaming_completion = await client.chat.completions.create(**request)
            
            streamed_chunks = []
            
            async for chunk in streaming_completion:
                if not ttft_logged:
                    ttft = time.time() - start_time
                    logger.info(f"LLM stream started (TTFT: {ttft:.2f}s, model: {request.get('model')})")
                    ttft_logged = True

                # Check for cancellation before yielding each chunk
                if request_id and request_id in self.active_requests:
                    if self.active_requests[request_id].is_set():
                        raise HTTPException(status_code=499, detail="Request cancelled by client")
                
                # Convert chunk to SSE format matching original HTTP client format
                chunk_dict = chunk.model_dump()
                streamed_chunks.append(chunk_dict)
                chunk_json = json.dumps(chunk_dict, ensure_ascii=False)
                yield f"data: {chunk_json}"
            
            duration = time.time() - start_time
            logger.info(f"LLM stream completed in {duration:.2f}s (model: {request.get('model')})")
            
            try:
                with open("last_openai_response_stream.json", "w", encoding="utf-8") as f:
                    json.dump(streamed_chunks, f, indent=2, ensure_ascii=False)
            except Exception as e_log:
                logger.error(f"Failed to write last_openai_response_stream.json: {e_log}")
            
            # Signal end of stream
            yield "data: [DONE]"
                
        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=self.classify_openai_error(str(e)))
        except RateLimitError as e:
            raise HTTPException(status_code=429, detail=self.classify_openai_error(str(e)))
        except BadRequestError as e:
            raise HTTPException(status_code=400, detail=self.classify_openai_error(str(e)))
        except APIError as e:
            status_code = getattr(e, 'status_code', 500)
            raise HTTPException(status_code=status_code, detail=self.classify_openai_error(str(e)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
        
        finally:
            # Clean up active request tracking
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    def classify_openai_error(self, error_detail: Any) -> str:
        """Provide specific error guidance for common OpenAI API issues."""
        error_str = str(error_detail).lower()
        
        # Region/country restrictions
        if "unsupported_country_region_territory" in error_str or "country, region, or territory not supported" in error_str:
            return "OpenAI API is not available in your region. Consider using a VPN or Azure OpenAI service."
        
        # API key issues
        if "invalid_api_key" in error_str or "unauthorized" in error_str:
            return "Invalid API key. Please check your OPENAI_API_KEY configuration."
        
        # Rate limiting
        if "rate_limit" in error_str or "quota" in error_str:
            return "Rate limit exceeded. Please wait and try again, or upgrade your API plan."
        
        # Model not found
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return "Model not found. Please check your BIG_MODEL and SMALL_MODEL configuration."
        
        # Billing issues
        if "billing" in error_str or "payment" in error_str:
            return "Billing issue. Please check your OpenAI account billing status."
        
        # Default: return original message
        return str(error_detail)
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request by request_id."""
        if request_id in self.active_requests:
            self.active_requests[request_id].set()
            return True
        return False
