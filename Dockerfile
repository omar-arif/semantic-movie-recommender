FROM python:3.12


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt

# install all requirements except sentence-transformers (in order to avoid large cuda modules)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


# install sentence transformers with no deps
RUN pip3 install --no-deps sentence-transformers==3.1.0


COPY ./app /code/app


CMD ["fastapi", "run", "app/app.py", "--port", "8080"]