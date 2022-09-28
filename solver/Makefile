test:
	pytest

test-1:
	pytest -v -m single

test-vv:
	pytest -vv -m verbose

daemon:
	tail -f /dev/null

dktest:
	docker build . -t handsolver
	docker run --rm -t handsolver test

dktest-1:
	docker build . -t handsolver
	docker run --rm -t handsolver test-1

dktest-vv:
	docker build . -t handsolver
	docker run --rm -t handsolver test-vv
