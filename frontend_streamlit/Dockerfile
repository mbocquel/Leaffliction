FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./src /code/

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501"]

