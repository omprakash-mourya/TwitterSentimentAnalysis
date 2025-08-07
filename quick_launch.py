"""
Quick Launch Script for Dark Theme Twitter Sentiment Analysis Dashboard
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def launch_dashboard():
    """Launch the dark theme dashboard with all fixes applied"""
    print("🌙 Launching Dark Theme Twitter Sentiment Analysis Dashboard")
    print("=" * 60)
    
    # Project directory
    project_dir = Path("c:/Users/ommou/OneDrive/Desktop/Twitter sentiment analysis/RealTimeSentimentApp")
    
    print(f"📁 Project directory: {project_dir}")
    print("🚀 Starting Streamlit with dark theme and chart fixes...")
    
    # Launch Streamlit
    port = 8506
    url = f"http://localhost:{port}"
    
    print(f"🌐 Dashboard will be available at: {url}")
    print("\n✅ Features enabled:")
    print("   🌙 Dark theme with professional styling")
    print("   📊 All charts optimized for dark backgrounds")
    print("   ⚡ GPU acceleration (RTX 3050 6GB)")
    print("   🔗 Connection timeout fixes")
    print("   📈 Large-scale analysis (up to 50K+ tweets)")
    
    try:
        # Change to project directory and run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app/streamlit_app.py", 
            "--server.port", str(port),
            "--server.headless", "false"
        ], cwd=project_dir, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
    except KeyboardInterrupt:
        print("\n⏹️ Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    launch_dashboard()
