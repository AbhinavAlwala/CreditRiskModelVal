.PHONY: all analysis sql test report clean

all: analysis sql test report

analysis:
	python run_analysis.py

sql:
	python run_sql_demo.py > outputs/sql_demo_output.txt

test:
	pytest -q

report:
	python generate_report.py

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
