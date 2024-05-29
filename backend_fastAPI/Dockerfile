FROM python:3.9

WORKDIR /code

COPY ./configs /code/configs
COPY ./dataloader /code/dataloader
COPY ./logs /code/logs
COPY ./model /code/model
COPY ./saved_models /code/saved_models
COPY ./utils /code/utils
COPY ./api.py /code/api.py
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["fastapi", "run", "api.py", "--port", "8888"]
