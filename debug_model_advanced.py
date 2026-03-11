import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

print("TensorFlow Version:", tf.__version__)
print("Keras Version:", tf.keras.__version__)

def test_strategy(name, func):
    print(f"\n--- Testing Strategy: {name} ---")
    try:
        model = func()
        print("SUCCESS.")
        if hasattr(model, 'input_shape'):
            print("Input Shape:", model.input_shape)
        return True
    except Exception as e:
        print("FAILURE.")
        print("Error:", e)
        return False

# Strategy 1: Weights=None, explicit shape
def strategy_1():
    print("Building with weights=None, input_shape=(224, 224, 3)...")
    return EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))

# Strategy 2: Weights='imagenet', NO input shape
def strategy_2():
    print("Building with weights='imagenet', input_shape=None...")
    return EfficientNetB0(weights='imagenet', include_top=False)

# Strategy 3: Functional API wrapping
def strategy_3():
    print("Building output-based model...")
    base_model = EfficientNetB0(weights='imagenet', include_top=False)
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    return tf.keras.Model(inputs, x)

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    
    # Check 1
    test_strategy("Weights=None", strategy_1)
    
    # Check 2
    tf.keras.backend.clear_session()
    test_strategy("No Input Shape", strategy_2)
    
    # Check 3
    tf.keras.backend.clear_session()
    test_strategy("Functional Wrap", strategy_3)
