# MLFlow Experiment Tracker Setup

## Issue

If you encounter the following error when running the ZenML E2E project:

```
RuntimeError: Your active stack needs to contain a MLFlow experiemnt tracker for this example to work
```

This error occurs because the active ZenML stack doesn't have an MLFlow experiment tracker configured.

## Solution

This repository includes a script to help you set up the required ZenML stack with MLFlow experiment tracker.

### Option 1: Run the setup script

```bash
python setup_mlflow_stack.py
```

This script will:
1. Check if the `make` command is available
2. Run the `make install-stack-local` command to set up the required ZenML stack
3. Provide feedback on the success or failure of the operation

### Option 2: Run the Makefile target directly

```bash
make install-stack-local
```

This command will:
1. Register an MLFlow experiment tracker
2. Register an MLFlow model registry
3. Register an MLFlow model deployer
4. Register an Evidently data validator
5. Create a stack with all these components
6. Set this stack as the active stack

## Verification

After running either of the above commands, you can verify that the MLFlow experiment tracker is configured in your active stack by running:

```bash
zenml stack describe
```

This should show that your active stack includes an MLFlow experiment tracker component.

## Running the Pipeline

Once the MLFlow experiment tracker is configured, you can run the pipeline without encountering the error:

```bash
python run.py
```

This will execute the training, deployment, and batch inference pipelines as configured in the `run.py` file.
