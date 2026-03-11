import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

print("TensorFlow Version:", tf.__version__)
print("Keras Version:", tf.keras.__version__)

try:
    print("Attempting to build EfficientNetB0 with 'imagenet' weights...")
    
    # Clear session
    tf.keras.backend.clear_session()
    tf.keras.backend.set_image_data_format('channels_last')
    
    model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    print("SUCCESS: Model built successfully.")
    print("Input Shape:", model.input_shape)
    
except Exception as e:
    print("\nFAILURE: Could not build model.")
    print("Error:", e)
    import traceback
    traceback.print_exc()
