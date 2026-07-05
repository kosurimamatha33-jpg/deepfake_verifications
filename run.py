import os
import sys
import subprocess

def run():
    print("=" * 60)
    print("           CertifyFace Web Application Launcher")
    print("=" * 60)
    
    # Path to virtualenv Python executable (Windows standard)
    venv_dir = "venv"
    if os.path.exists(os.path.join(venv_dir, "Scripts", "python.exe")):
        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
        pip_exe = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        python_exe = sys.executable
        pip_exe = "pip"
        print("WARNING: Virtual environment not found or active. Using system Python.")
        
    print(f"Using Python: {python_exe}")
    
    # Check dependencies
    print("\n[1/2] Verifying Python dependencies...")
    try:
        import fastapi
        import uvicorn
        print("All dependencies (FastAPI, Uvicorn) are already installed.")
    except ImportError:
        print("Dependencies missing. Installing from requirements.txt...")
        try:
            subprocess.check_call([pip_exe, "install", "-r", "requirements.txt"])
            print("Dependencies installed successfully.")
        except Exception as e:
            print(f"ERROR: Failed to install dependencies: {e}")
            sys.exit(1)
            
    # Start server
    print("\n[2/2] Launching FastAPI Web Application...")
    print("Point your browser to: http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        # Run uvicorn server: main:app
        subprocess.check_call([python_exe, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"])
    except KeyboardInterrupt:
        print("\nServer stopped by operator.")
    except Exception as e:
        print(f"ERROR: Server failed to start: {e}")

if __name__ == "__main__":
    run()
