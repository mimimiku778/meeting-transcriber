"""Meeting Transcriber MCP Server - Transcribe meeting recordings with speaker diarization."""

# IMPORTANT: Set environment variables before any torch imports
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

__version__ = "0.1.0"
