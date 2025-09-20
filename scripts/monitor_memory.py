#!/usr/bin/env python3
"""
Memory monitoring script for training.
"""

import torch
import psutil
import time
import os

def get_memory_info():
    """Get current memory usage information."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        gpu_memory_free = gpu_memory_total - gpu_memory_reserved
    else:
        gpu_memory = gpu_memory_reserved = gpu_memory_total = gpu_memory_free = 0
    
    # CPU memory
    cpu_memory = psutil.virtual_memory()
    cpu_used = cpu_memory.used / 1024**3  # GB
    cpu_total = cpu_memory.total / 1024**3  # GB
    cpu_free = cpu_memory.available / 1024**3  # GB
    
    return {
        'gpu_used': gpu_memory,
        'gpu_reserved': gpu_memory_reserved,
        'gpu_total': gpu_memory_total,
        'gpu_free': gpu_memory_free,
        'cpu_used': cpu_used,
        'cpu_total': cpu_total,
        'cpu_free': cpu_free,
    }

def print_memory_status():
    """Print current memory status."""
    mem = get_memory_info()
    
    print("=" * 60)
    print("🔍 MEMORY STATUS")
    print("=" * 60)
    print(f"GPU Memory:")
    print(f"  Used: {mem['gpu_used']:.2f} GB / {mem['gpu_total']:.2f} GB ({mem['gpu_used']/mem['gpu_total']*100:.1f}%)")
    print(f"  Reserved: {mem['gpu_reserved']:.2f} GB")
    print(f"  Free: {mem['gpu_free']:.2f} GB")
    print(f"CPU Memory:")
    print(f"  Used: {mem['cpu_used']:.2f} GB / {mem['cpu_total']:.2f} GB ({mem['cpu_used']/mem['cpu_total']*100:.1f}%)")
    print(f"  Free: {mem['cpu_free']:.2f} GB")
    print("=" * 60)
    
    # Memory warnings
    if mem['gpu_free'] < 1.0:  # Less than 1GB free
        print("⚠️  WARNING: Low GPU memory!")
    if mem['cpu_free'] < 2.0:  # Less than 2GB free
        print("⚠️  WARNING: Low CPU memory!")

def clear_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU memory cleared")

if __name__ == "__main__":
    print_memory_status()
    
    # Optionally clear memory
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--clear":
        clear_memory()
        print_memory_status()
