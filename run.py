#!/usr/bin/env python3
"""
Quick start script for the Twitter Sentiment Analysis Dashboard
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the Streamlit app
    app_path = os.path.join(script_dir, "app", "streamlit_app.py")
    
    # Check if the app file exists
    if not os.path.exists(app_path):
        print(f"Error: Could not find {app_path}")
        sys.exit(1)
    
    # Run the Streamlit app
    try:
        print("🚀 Starting Twitter Sentiment Analysis Dashboard...")
        print("📱 Your browser should open automatically")
        print("🔗 If not, visit: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Thanks for using the Twitter Sentiment Analysis Dashboard!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running the application: {e}")
        print("💡 Make sure you have installed all requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
