FROM apache/airflow:2.7.2-python3.11

USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional ML/NLP packages
RUN pip install --no-cache-dir \
    scikit-learn==1.3.0 \
    nltk==3.8.1 \
    spacy==3.6.1 \
    great-expectations==0.18.15 \
    boto3==1.26.137 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy Airflow configuration and DAGs
COPY airflow.cfg /opt/airflow/airflow.cfg
COPY dags/ /opt/airflow/dags/
COPY plugins/ /opt/airflow/plugins/

# Set ownership
USER airflow

EXPOSE 8080

CMD ["airflow", "standalone"]
