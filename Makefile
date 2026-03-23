PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit

SPACE ?= amithjkamath/rtssdiffviewer
HF_REMOTE ?= hf
DEPLOY_BRANCH ?= deploy

.PHONY: install run fmt lint clean deploy-init deploy status

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(STREAMLIT) run app.py

fmt:
	@echo "No formatter configured."

lint:
	@if [ -x "$(VENV)/bin/python" ]; then \
		$(VENV)/bin/python -m py_compile app.py src/rtssdiffviewer/*.py; \
	else \
		$(PYTHON) -m py_compile app.py src/rtssdiffviewer/*.py; \
	fi

clean:
	rm -rf $(VENV) __pycache__ src/rtssdiffviewer/__pycache__

deploy-init:
	@if git remote | grep -q '^$(HF_REMOTE)$$'; then \
		git remote set-url $(HF_REMOTE) https://huggingface.co/spaces/$(SPACE); \
	else \
		git remote add $(HF_REMOTE) https://huggingface.co/spaces/$(SPACE); \
	fi
	@git fetch $(HF_REMOTE) || true
	@if git show-ref --verify --quiet refs/heads/$(DEPLOY_BRANCH); then \
		echo "Deploy branch $(DEPLOY_BRANCH) already exists"; \
	else \
		git checkout -b $(DEPLOY_BRANCH); \
		git checkout -; \
	fi

deploy:
	bash deploy.sh $(SPACE)

status:
	@echo "Space URL: https://huggingface.co/spaces/$(SPACE)"
