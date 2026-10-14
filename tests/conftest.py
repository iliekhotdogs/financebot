"""Pytest/unittest conftest — loads .env and sets up test path."""
import os
import sys

# Add finance-bot root to path
BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOT_DIR)

# Load environment variables from .env (credentials, etc.)
from dotenv import load_dotenv
load_dotenv(os.path.join(BOT_DIR, ".env"))
