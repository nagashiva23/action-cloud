.PHONY: help setup up down reset api worker verify test logs psql clean

help:
	@echo "ActionCloud — Phase 1"
	@echo ""
	@echo "  make setup     create venv + install dependencies"
	@echo "  make up        start Postgres + LocalStack"
	@echo "  make down      stop containers (keeps data)"
	@echo "  make reset     stop AND wipe data (re-runs sql/001_init.sql)"
	@echo "  make api       run the Agent Memory API   (terminal 1)"
	@echo "  make worker    run the queue worker       (terminal 2)"
	@echo "  make verify    run end-to-end verification (terminal 3)"
	@echo "  make test      run unit tests"
	@echo "  make logs      tail container logs"
	@echo "  make psql      open a psql shell"

setup:
	python3 -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "\nDone. Activate with:  source .venv/bin/activate"

up:
	docker compose up -d
	@echo "waiting for services to report healthy..."
	@sleep 8
	@docker compose ps

down:
	docker compose down

# Use this after editing sql/001_init.sql — the init script only runs on a
# fresh volume, so without -v your schema changes are silently ignored.
reset:
	docker compose down -v
	docker compose up -d
	@sleep 8
	@docker compose ps

api:
	PYTHONPATH=src ./.venv/bin/uvicorn actioncloud.api:app --reload --port 8000

worker:
	PYTHONPATH=src ./.venv/bin/python -m actioncloud.worker

verify:
	./.venv/bin/python scripts/verify_e2e.py

test:
	PYTHONPATH=src ./.venv/bin/pytest tests/ -v

logs:
	docker compose logs -f

psql:
	docker compose exec postgres psql -U actioncloud -d actioncloud

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
