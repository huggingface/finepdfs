import asyncio
import json
import os
import signal
import time
import uuid
import argparse
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, ModernBertForSequenceClassification
from multi_head_moder_bert import MultiHeadModernBert

from torch.nn import Module
torch.set_float32_matmul_precision('high')


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionRequest(BaseModel):
    input: str  # List of text inputs for classification
    max_tokens: int = 100
    temperature: float = 0.0
    model: Optional[str] = None

class CompletionResult(BaseModel):
    text: str  # Raw logits/scores for each class
    usage: Usage
    finish_reason: str = "stop"

class CompletionResponse(BaseModel):
    id: str
    object: str = "completion"
    created: int
    model: str
    choices: List[CompletionResult]

# Global variables for server state
transformers_model: Optional[AutoModelForSequenceClassification] = None
transformers_tokenizer: Optional[AutoTokenizer] = None
pending_classification_requests: List[Dict[str, Any]] = []
classification_futures: Dict[str, asyncio.Future] = {}
classification_batch_lock = asyncio.Lock()
last_classification_batch_time: Optional[float] = None
classification_timeout_task: Optional[asyncio.Task] = None
shutting_down = False
shutdown_event = asyncio.Event()

# Configuration - will be set via CLI arguments
config = {}

def _prepare_classification_inputs(requests: List[Dict[str, Any]]) -> List[str]:
    """Prepare text inputs from batched classification requests."""
    text_inputs = []
    for req_data in requests:
        request = req_data["request"]
        # Each request can have multiple input texts
        text_inputs.append(request.input)
    
    return text_inputs

@torch.no_grad()
def run_classification_inference(current_batch: List[Dict[str, Any]]) -> List[Union[float, int]]:
    """Run classification inference and return label predictions."""
    global transformers_model, transformers_tokenizer, config
    
    batch_size = len(current_batch)
    logger.info(f"Running classification inference for {batch_size} requests")
    
    # Prepare text inputs
    text_inputs = _prepare_classification_inputs(current_batch)
    
    # Tokenize inputs
    inputs = transformers_tokenizer(
        text_inputs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=config["max_context"],  # Reasonable limit for classification
    )

    input_lengths = [len(l) for l in inputs["input_ids"].tolist()]
    
    # Move to device
    device = next(transformers_model.parameters()).device
    inputs = inputs.to(device)
    
    # Get model predictions
    with torch.inference_mode():
        outputs = transformers_model(**inputs)
        logits = outputs.logits.cpu().tolist()
    
    return logits, input_lengths

async def _process_classification_batch(force_partial: bool = False):
    """Process batch_size classification requests."""
    global pending_classification_requests, classification_futures, last_classification_batch_time, shutting_down
    
    if shutting_down:
        return
        
    async with classification_batch_lock:
        if shutting_down or len(pending_classification_requests) == 0:
            return
            
        batch_size = config["batch_size"]
        if not force_partial and len(pending_classification_requests) < batch_size:
            return
        
        current_batch = pending_classification_requests[:min(batch_size, len(pending_classification_requests))]
        logger.info(f"Processing classification batch of {len(current_batch)} requests")
        
        try:
            if transformers_model is None or transformers_tokenizer is None:
                raise RuntimeError("Transformers model not initialized")
            
            # Get all text inputs and track which belong to which request
            all_predictions, all_input_lengths = run_classification_inference(current_batch)
            
            # Remove processed requests from pending list
            pending_classification_requests = pending_classification_requests[len(current_batch):]
            last_classification_batch_time = time.time()
            
        except Exception as e:
            logger.error(f"Error processing classification batch: {e}")
            # Resolve futures with error
            for req_data in current_batch:
                request_id = req_data["request_id"]
                if request_id in classification_futures:
                    classification_futures[request_id].set_exception(e)
                    del classification_futures[request_id]
            return
            
    # Create responses for each request
    for pred_idx, req_data in enumerate(current_batch):
        request_id = req_data["request_id"]
        request = req_data["request"]
        
        # Extract predictions for this request's inputs
        request_predictions = all_predictions[pred_idx]
        request_input_lengths = all_input_lengths[pred_idx]
        
        # Create classification results
        classification_result = CompletionResult(
            # Big meh, but it's only way to not bloat datatrove
            text=",".join([str(x) for x in request_predictions]),
            usage=Usage(
                prompt_tokens=request_input_lengths,
                completion_tokens=1,
                total_tokens=request_input_lengths + 1
            )
        )
        
        response = CompletionResponse(
            id=f"classify-{uuid.uuid4().hex[:29]}",
            created=int(time.time()),
            model=request.model or config["model_name_or_path"],
            choices=[classification_result]
        )
        
        if request_id in classification_futures:
            classification_futures[request_id].set_result(response)
            del classification_futures[request_id]

async def _classification_timeout_checker():
    """Background task to check for classification batch timeouts."""
    global last_classification_batch_time, pending_classification_requests, shutting_down
    
    while not shutting_down:
        try:
            await asyncio.sleep(0.1)  # Check more frequently
            
            if shutting_down:
                break
                
            current_time = time.time()
            batch_timeout = config["batch_timeout"]
            
            # Process partial batches if we have pending requests and enough time has passed
            if (pending_classification_requests and 
                (last_classification_batch_time is None or 
                 current_time - last_classification_batch_time >= batch_timeout)):
                
                logger.info(f"Timeout checker processing partial batch of {len(pending_classification_requests)} requests")
                await _process_classification_batch(force_partial=True)
                
        except asyncio.CancelledError:
            logger.info("Classification timeout checker cancelled")
            break
        except Exception as e:
            logger.error(f"Error in classification timeout checker: {e}", exc_info=True)
            # Continue running even if there's an error

def graceful_shutdown():
    """Handle graceful shutdown of the server."""
    global shutting_down, classification_timeout_task, pending_classification_requests, classification_futures
    
    logger.info("Starting graceful shutdown...")
    shutting_down = True
    
    # Cancel timeout tasks
    if classification_timeout_task and not classification_timeout_task.done():
        classification_timeout_task.cancel()
    
    # Cancel any remaining futures
    if pending_classification_requests:
        logger.info(f"Cancelling {len(pending_classification_requests)} unprocessed classification requests")
        for req_data in pending_classification_requests:
            request_id = req_data["request_id"]
            if request_id in classification_futures:
                classification_futures[request_id].cancel()
                del classification_futures[request_id]
        pending_classification_requests.clear()
    
    shutdown_event.set()
    logger.info("Graceful shutdown completed")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    graceful_shutdown()

def create_app(
    model_name_or_path: str,
    batch_size: int = 4,
    batch_timeout: float = 10.0,
    max_context: int = 2048,
    model_kwargs: Optional[Dict[str, Any]] = None
) -> FastAPI:
    """Create the FastAPI application with CLI-provided configuration."""
    global config
    
    # Set configuration from CLI arguments
    config = {
        "model_name_or_path": model_name_or_path,
        "batch_size": batch_size,
        "batch_timeout": batch_timeout,
        "max_context": max_context,
        "model_kwargs": model_kwargs or {}
    }
    
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    app = FastAPI(
        title="Transformers Classification Batching Server",
        description="Classification API with Transformers batching for sequence classification models"
    )

    @app.on_event("startup")
    async def startup_event():
        """Initialize Transformers model and start timeout checker on startup."""
        global transformers_model, transformers_tokenizer, last_classification_batch_time, classification_timeout_task
        
        logger.info(f"Initializing Transformers classification model: {config['model_name_or_path']}")
        
        # Default kwargs for classification models
        model_kwargs = {
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2"
        }
        
        if config.get("model_kwargs"):
            model_kwargs.update(config["model_kwargs"])
        
        # Initialize model and tokenizer
        model_names = config["model_name_or_path"].split(";")
        if len(model_names) == 1:
            transformers_model = AutoModelForSequenceClassification.from_pretrained(
                config["model_name_or_path"],
                **model_kwargs,
                num_labels=1
            )
        else:
            models = [
                AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    **model_kwargs,
                    num_labels=1
                )
                for model_name in model_names
            ]
            transformers_model = MultiHeadModernBert(models)
            del models

        # Move to GPU
        transformers_model.to("cuda")

        # Force python garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # transformers_model.compile(mode="reduce-overhead", fullgraph=True)

        
        transformers_tokenizer = AutoTokenizer.from_pretrained(model_names[0])
        
        logger.info("Transformers classification model initialized successfully")

        # Run a test inference with config["batch_size"] requests and max_context length
        logger.info("Running test inference to warm up the model...")
        test_texts = ["This is a test sentence for model warm-up."*2048] * config["batch_size"]
        test_requests = [CompletionRequest(input=text) for text in test_texts]
        test_pending_requests = [
            {"request_id": f"test_{i}", "request": request}
            for i, request in enumerate(test_requests)]

        run_classification_inference(test_pending_requests)
        logger.info("Test inference completed successfully")
        
        # Initialize the last batch time to None so timeout checker can process immediately
        last_classification_batch_time = None
        classification_timeout_task = asyncio.create_task(_classification_timeout_checker())
        logger.info(f"Started classification batch timeout checker with {config['batch_timeout']}s timeout")

    @app.post("/v1/completions")
    async def complete(request: CompletionRequest) -> CompletionResponse:
        """Handle classification requests with batching."""
        global pending_classification_requests, classification_futures, shutting_down
        
        if shutting_down:
            raise HTTPException(status_code=503, detail="Server is shutting down")
        
        request_id = uuid.uuid4().hex
        future = asyncio.Future()
        classification_futures[request_id] = future
        
        pending_classification_requests.append({
            "request_id": request_id,
            "request": request
        })
        logger.info(f"Received classification request {request_id}, batch size: {len(pending_classification_requests)}")
        
        if len(pending_classification_requests) >= config["batch_size"]:
            logger.info(f"Batch size reached ({len(pending_classification_requests)}), processing immediately")
            await _process_classification_batch()
        else:
            logger.debug(f"Request queued, waiting for batch. Current size: {len(pending_classification_requests)}/{config['batch_size']}")
        
        try:
            response = await future
            return response
        except Exception as e:
            logger.error(f"Error processing classification request {request_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        if shutting_down:
            raise HTTPException(status_code=503, detail="Server is shutting down")
        return {"status": "healthy"}

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [
                {
                    "id": config["model_name_or_path"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "transformers"
                }
            ]
        }

    return app

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Transformers Classification Batching Server")
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Model name or path for the transformers classification model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)"
    )
    parser.add_argument(
        "--batch-timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds for partial batch processing (default: 10.0)"
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=2048,
        help="Maximum context length for classification (default: 2048)"
    )
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default="{}",
        help="JSON string of additional model kwargs (default: '{}')"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Parse model kwargs
    try:
        model_kwargs = json.loads(args.model_kwargs)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON for model-kwargs: {args.model_kwargs}")
        return 1
    
    # Create the app with CLI arguments
    app = create_app(
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        batch_timeout=args.batch_timeout,
        max_context=args.max_context,
        model_kwargs=model_kwargs
    )
    
    # Start the server
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    import sys
    sys.exit(main())

# For external usage (when imported)
app = None  # Will be None unless create_app is called directly
