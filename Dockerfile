FROM python:3.8

COPY . .

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

CMD streamlit run app.py --server.port 8501 --server.enableXsrfProtection=false