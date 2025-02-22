# Use the Python base image
FROM python:3.13
 
COPY --from=openjdk:8-jre-slim /usr/local/openjdk-8 /usr/local/openjdk-8
ENV JAVA_HOME=/usr/local/openjdk-8
RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk-8/bin/java 1
 
# Create and set the working directory
WORKDIR /app
 
# Copy the requirements file and install dependencies
COPY . /app
RUN pip3 install -r requirements.txt --no-cache-dir

# Install decision-rules from github
RUN git clone https://github.com/ruleminer/decision-rules.git
RUN pip3 install ./decision-rules
 
RUN python -m rulekit download_jar

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "rulekit_gui_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
