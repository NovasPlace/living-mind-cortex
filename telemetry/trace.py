"""
Telemetry — Sovereign Software AgentRuntime
Dynamically instruments and wraps every single public module method 
to emit deterministic websocket payload traces.
"""

import asyncio
import inspect
import json
from api.events import manager

def _emit(organ_name: str, func_name: str):
    """Fire-and-forget payload emission on the running event loop."""
    try:
        loop = asyncio.get_running_loop()
        message = json.dumps({
            "organ": organ_name,
            "function": func_name
        })
        loop.create_task(manager.broadcast_event("function_fire", message))
    except RuntimeError:
        pass

def wrap_function(organ_name: str, func_name: str, original_func):
    """Wraps functions while deeply preserving their sync/async signatures"""
    
    if asyncio.iscoroutinefunction(original_func):
        async def async_wrapper(*args, **kwargs):
            _emit(organ_name, func_name)
            return await original_func(*args, **kwargs)
        return async_wrapper
    else:
        def sync_wrapper(*args, **kwargs):
            _emit(organ_name, func_name)
            return original_func(*args, **kwargs)
        return sync_wrapper

def inject_telemetry(instances_dict: dict):
    """
    Given a dict of { organ_name: instance }, scan every public method and overwrite.
    """
    for organ_name, instance in instances_dict.items():
        # Get all public attributes that aren't dunder methods
        noise_filters = {"stats", "census", "report", "inflammation", "brain_rate", "pulse_scale"}
        for attr_name in dir(instance):
            if attr_name.startswith("_") or attr_name in noise_filters:
                continue
            
            attr = getattr(instance, attr_name)
            
            # We exclusively want to trace callable methods bound to the instance
            if inspect.ismethod(attr):
                wrapped = wrap_function(organ_name, attr_name, attr)
                # Overwrite on the object
                setattr(instance, attr_name, wrapped)
                
    print("[TELEMETRY] Hyper-Visor Hooked. All organs are now omnisciently traced.")
