FROM python:3.12


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt

# install all requirements except sentence-transformers (in order to avoid large cuda modules)
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


# install sentence transformers with no deps
RUN pip3 install --no-deps sentence-transformers==3.1.0


# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/code

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user ./app $HOME/code/app


CMD ["fastapi", "run", "app/app.py", "--port", "7860"]