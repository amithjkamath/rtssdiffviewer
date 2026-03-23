PYTHON ?= python3
UV ?= uv
PYTHON_VERSION ?= 3.11
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
STREAMLIT := $(VENV)/bin/streamlit

SPACE ?= amithjkamath/rtssdiffviewer
HF_REMOTE ?= hf
DEPLOY_BRANCH ?= deploy

.PHONY: check-uv venv install run test test-all fmt lint clean clean-venv clean-py deploy-init deploy deploy-now status

check-uv:
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "uv is required. Install from https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1; \
	}

venv:
	$(MAKE) check-uv
	rm -rf $(VENV)
	$(UV) venv --python $(PYTHON_VERSION) $(VENV)

install:
	$(MAKE) venv
	$(UV) pip install --python $(VENV)/bin/python -r requirements.txt
	$(UV) pip install --python $(VENV)/bin/python -e .

run:
	$(STREAMLIT) run app.py

test:
	@if [ -x "$(PYTEST)" ]; then \
		$(PYTEST) -q; \
	else \
		echo "pytest is not installed in $(VENV). Run 'make install' first."; \
		exit 1; \
	fi

test-all:
	$(MAKE) install
	$(MAKE) test

fmt:
	@echo "No formatter configured."

lint:
	@if [ -x "$(VENV)/bin/python" ]; then \
		$(VENV)/bin/python -m py_compile app.py src/rtssdiffviewer/*.py; \
	else \
		$(PYTHON) -m py_compile app.py src/rtssdiffviewer/*.py; \
	fi

clean:
	$(MAKE) clean-venv
	$(MAKE) clean-py

clean-venv:
	rm -rf $(VENV)

clean-py:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov .ipynb_checkpoints

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

deploy-now:
	$(MAKE) deploy-init
	@git add -A && git commit -m "Deploy: $(shell date '+%Y-%m-%d %H:%M:%S')" || true
	@git push $(HF_REMOTE) $(DEPLOY_BRANCH):main
	@echo "Deployment complete. Space: https://huggingface.co/spaces/$(SPACE)"

status:
	@echo "Space URL: https://huggingface.co/spaces/$(SPACE)"
