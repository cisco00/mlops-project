"""
This script helps set up the required ZenML stack with MLFlow experiment tracker.

The error "RuntimeError: Your active stack needs to contain a MLFlow experiemnt tracker for
this example to work" occurs because the active ZenML stack doesn't have an MLFlow
experiment tracker configured.

This script will run the necessary commands to set up the required stack.
"""

import os
import subprocess
import sys

def main():
    print("Setting up ZenML stack with MLFlow experiment tracker...")

    # Check if make is installed
    try:
        subprocess.run(["make", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'make' command not found. Please install make and try again.")
        sys.exit(1)

    # Run the install-stack-local target from the Makefile
    try:
        # Set the stack name environment variable
        stack_name = "e2e_template_stack"  # Default stack name
        env = os.environ.copy()
        env["stack_name"] = stack_name

        print(f"Setting up ZenML stack with name: {stack_name}")

        # Capture the output of the subprocess
        result = subprocess.run(["make", "install-stack-local"], check=True, capture_output=True, text=True, env=env)

        # Print the output for debugging
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)

        print("\nSuccess! The ZenML stack with MLFlow experiment tracker has been set up.")
        print("You can now run your pipeline without the error.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to set up the ZenML stack. Error code: {e.returncode}")
        print("Error output:")
        print(e.stderr)
        print("\nPlease try running 'make install-stack-local' manually.")
        sys.exit(1)

if __name__ == "__main__":
    main()