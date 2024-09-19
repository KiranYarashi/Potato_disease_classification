import tensorflow as tf

# Load your .keras model
model = tf.keras.models.load_model('saved_models/1/1.keras')

# Export the model in SavedModel format
model.export('saved_models/a/')

# Repeat for other models
model = tf.keras.models.load_model('saved_models/2/2.keras')
model.export('saved_models/b/')
