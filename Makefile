.DEFAULT_GOAL := push

DEFAULT_MSG := Commit per build, to track
MSG ?= $(DEFAULT_MSG)

.PHONY: push help

push:
	git add * .gitignore
	git commit -m "$(MSG)"
	git push
	@echo "Commit/push on build complete"

help:
	@echo Usage:
	@echo   make push
	@echo   make push MSG=\"commit message\"
	@echo   make help
	@echo Behavior:
	@echo   - Stages all files (*) and .gitignore
	@echo   - Commits with the provided message or a default message
	@echo   - Pushes to the current remote
