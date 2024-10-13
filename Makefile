# Define the virtual environment directory
VENV_DIR = venv

# Define the Python interpreter and pip for the virtual environment
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

# Define the requirements file
REQUIREMENTS = requirements.txt

# Default target
all: install

# Create a virtual environment and install dependencies
install:
	python3 -m venv $(VENV_DIR)
	$(PIP) install -r $(REQUIREMENTS)

# Run the application
run:
	$(PYTHON) -m flask run --host=0.0.0.0 --port=3000

# Clean up the virtual environment
clean:
	rm -rf $(VENV_DIR)
(VENV)/bin/activate && FLASK_APP=$(APP_NAME) flask run --host=0.0.0.0 --port=$(PORT)