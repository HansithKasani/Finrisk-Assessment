"""
Launcher script for Credit Risk Assessment Streamlit Application
Convenient way to start the application
"""

import os
import sys
from pathlib import Path
import subprocess

def main():
    """Launch the Streamlit application"""
    
    # Get the App directory
    app_dir = Path(__file__).parent / "App"
    main_file = app_dir / "main.py"
    
    if not main_file.exists():
        print("❌ Error: App/main.py not found!")
        print(f"Expected location: {main_file}")
        sys.exit(1)
    
    print("🚀 Launching Credit Risk Assessment Dashboard...")
    print(f"📂 App location: {main_file}")
    print("🌐 Opening browser at http://localhost:8501")
    print("\n" + "="*60)
    print("Press Ctrl+C to stop the application")
    print("="*60 + "\n")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(main_file),
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n\n✅ Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
