# Use the official TensorFlow Serving image
FROM tensorflow/serving

# Copy the models and config file into the Docker image
COPY saved_models /models/saved_models
COPY models.config /models/models.config

# Specify the configuration file for TensorFlow Serving
ENTRYPOINT ["tensorflow_model_server", "--port=8500", "--rest_api_port=8501", "--model_config_file=/models/models.config"]
