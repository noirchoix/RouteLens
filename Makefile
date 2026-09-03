.PHONY: install test test-log ui-install ui-check ui-build freeze context benchmark map derive indexes train validate lock product-two serve clean-v2

install:
	python -m pip install -e ".[dev,mapping]"

test:
	pytest

test-log:
	python scripts/run_pytest_logged.py

ui-install:
	cd ui && npm install

ui-check:
	cd ui && npm run check

ui-build:
	cd ui && npm run build:embed

freeze:
	reacts --project-root . freeze-product-one

context:
	reacts --project-root . build-contextual-v2 --resume

benchmark:
	reacts --project-root . benchmark-mapper --backend rxnmapper --batch-sizes 8 16 32 64 --sample-size 512

map:
	reacts --project-root . map-reactions --backend rxnmapper --fallback-backend mcs --batch-size 16 --workers 1 --shard-size 5000 --resume

derive:
	reacts --project-root . derive-reaction-centres --resume

indexes:
	reacts --project-root . build-product-two-indexes --resume

train:
	reacts --project-root . train-product-two --request-promotion

validate:
	reacts --project-root . validate-product-two

lock:
	reacts --project-root . lock-product-two --release-id v2.0.2

product-two:
	reacts --project-root . product-two --mapping-backend rxnmapper --batch-size 16 --resume --request-promotion

serve:
	reacts --project-root . serve --port 8000

clean-v2:
	rm -rf data/canonical_v2_context data/mapping_v2 data/derivation_v2 data/canonical_v2 data/indexes_v2 data/state/product_two_mapping.sqlite3* data/state/product_two_derivation.sqlite3* data/releases/v2.0.2 reports/product_two_*.json reports/mapping
