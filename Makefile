# Variables
BACKEND_DIR = backend
FRONTEND_DIR = frontend/favorite-movie-app
PYTHON = python3
PIP = pip3
NODE = node
NPM = npm

# Docker variables
DOCKER_COMPOSE = docker compose
DOCKER_BUILD = docker build

# Colors for terminal output
GREEN = \033[0;32m
NC = \033[0m # No Color
RED = \033[0;31m

.PHONY: all install clean run build test up down docker-build verify-models rebuild-models

# Default target
all: install build

# Docker commands for primary app control
up: docker-build
	@echo "${GREEN}Starting application containers...${NC}"
	$(DOCKER_COMPOSE) up -d
	@echo "${GREEN}Application is running${NC}"

down:
	@echo "Stopping application containers..."
	$(DOCKER_COMPOSE) down
	@echo "${GREEN}Application stopped${NC}"

# Installation commands
install: install-backend install-frontend
	@echo "${GREEN}All dependencies installed${NC}"

install-backend:
	@echo "Installing backend dependencies..."
	cd $(BACKEND_DIR) && $(PIP) install -r requirements.txt

install-frontend:
	@echo "Installing frontend dependencies..."
	cd $(FRONTEND_DIR) && $(NPM) install

# Build commands
build: build-frontend
	@echo "${GREEN}Build complete${NC}"

build-frontend:
	@echo "Building frontend..."
	cd $(FRONTEND_DIR) && $(NPM) run build

# Run commands (for development without Docker)
run: run-backend run-frontend
	@echo "${GREEN}Application is running${NC}"

run-backend:
	@echo "Starting backend server..."
	cd $(BACKEND_DIR) && $(PYTHON) app.py

run-frontend:
	@echo "Starting frontend development server..."
	cd $(FRONTEND_DIR) && $(NPM) start

# Test commands
test: test-backend test-frontend
	@echo "${GREEN}All tests complete${NC}"

test-backend:
	@echo "Running backend tests..."
	cd $(BACKEND_DIR) && $(PYTHON) -m pytest

test-frontend:
	@echo "Running frontend tests..."
	cd $(FRONTEND_DIR) && $(NPM) test

# Docker build command
docker-build: verify-models
	@echo "Building Docker images..."
	@echo "Ensuring data directories exist..."
	mkdir -p $(BACKEND_DIR)/data/model_cache
	mkdir -p $(BACKEND_DIR)/data/models
	docker build -t movie-app-backend ./$(BACKEND_DIR) || (echo "${RED}Backend build failed${NC}" && exit 1)
	docker build -t movie-app-frontend ./$(FRONTEND_DIR) || (echo "${RED}Frontend build failed${NC}" && exit 1)

# Add these new commands
verify-models:
	@echo "Verifying model files..."
	@cd $(BACKEND_DIR) && $(PYTHON) scripts/verify_models.py || ( \
		echo "${RED}Some model files are invalid or missing. Attempting to rebuild...${NC}" && \
		$(MAKE) rebuild-models \
	)

rebuild-models:
	@echo "${GREEN}Rebuilding model files...${NC}"
	@cd $(BACKEND_DIR) && $(PYTHON) scripts/rebuild_models.py || ( \
		echo "${RED}Failed to rebuild models${NC}" && \
		exit 1 \
	)

# Clean commands
clean: clean-backend clean-frontend down
	@echo "${GREEN}Clean complete${NC}"

clean-backend:
	@echo "Cleaning backend..."
	find $(BACKEND_DIR) -type d -name "__pycache__" -exec rm -rf {} +
	find $(BACKEND_DIR) -type f -name "*.pyc" -delete

clean-frontend:
	@echo "Cleaning frontend..."
	cd $(FRONTEND_DIR) && rm -rf node_modules build

# Development helper commands
dev: install
	@echo "Starting development servers..."
	make -j 2 run-backend run-frontend

# Help command
help:
	@echo "Available commands:"
	@echo "  make up          - Start the application with Docker"
	@echo "  make down        - Stop and remove Docker containers"
	@echo "  make install     - Install all dependencies"
	@echo "  make build       - Build the application"
	@echo "  make run         - Run both backend and frontend (development)"
	@echo "  make test        - Run all tests"
	@echo "  make clean       - Clean build artifacts and stop containers"
	@echo "  make dev         - Start development environment"
	@echo "  make docker-build - Build Docker images"