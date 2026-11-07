setup:
	pip install -r requirements.txt
run:
	streamlit run app.py
report:
	python -m src.report --lang ja && python -m src.report --lang en
evaluate:
	python -m src.evaluate
test:
	pytest -q
