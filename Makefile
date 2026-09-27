.PHONY: help setup up down reset migrate agent api worker verify test logs psql mcp experiment fleet calibrate clean

help:
	@echo "ActionCloud"
	@echo ""
	@echo "  make setup     create venv + install dependencies"
	@echo "  make up        start Postgres + LocalStack"
	@echo "  make down      stop containers (keeps data)"
	@echo "  make reset     stop AND wipe data (re-runs every sql/*.sql)"
	@echo "  make migrate   apply sql/002+ to an EXISTING database volume"
	@echo "  make agent ID=.. ROLE=..  register an agent, print its API key"
	@echo "  make api       run the Agent Memory API   (terminal 1)"
	@echo "  make worker    run the queue worker       (terminal 2)"
	@echo "  make verify    run end-to-end verification (terminal 3)"
	@echo "  make test      run unit tests"
	@echo "  make logs      tail container logs"
	@echo "  make psql      open a psql shell"
	@echo "  make mcp       run the MCP stdio server"
	@echo "  make experiment  run all evaluation presets (5 seeds) -> results/"
	@echo "  make fleet     run the 12-role fleet simulation"
	@echo "  make calibrate print similarity-threshold calibration"

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

# Every migration after 001 is idempotent, so re-running is always safe.
migrate:
	@for f in sql/0*.sql; do \
	  [ "$$f" = "sql/001_init.sql" ] && continue; \
	  echo "applying $$f"; \
	  docker compose exec -T postgres psql -q -U actioncloud -d actioncloud -v ON_ERROR_STOP=1 < $$f || exit 1; \
	done

# Register an agent and print its API key once:  make agent ID=cursor-1 ROLE=coding
agent:
	@test -n "$(ID)" -a -n "$(ROLE)" || (echo "usage: make agent ID=<agent_id> ROLE=<role>"; exit 2)
	PYTHONPATH=src ./.venv/bin/python -m actioncloud.auth create $(ID) $(ROLE)

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

mcp:
	PYTHONPATH=src ./.venv/bin/python -m actioncloud.mcp_server

experiment:
	PYTHONPATH=src ./.venv/bin/python scripts/run_experiment.py --preset all --seeds 1,2,3,4,5

fleet:
	PYTHONPATH=src ./.venv/bin/python scripts/run_fleet_simulation.py

calibrate:
	PYTHONPATH=src ./.venv/bin/python scripts/calibrate_threshold.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
