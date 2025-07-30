import joblib
import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
from utils import load_data
from sklearn.metrics import r2_score, mean_squared_error

def min_max_quantize(arr):
    arr_min, arr_max = arr.min(), arr.max()
    
    if arr_max == arr_min:
        quantized = np.full(arr.shape, 127, dtype=np.uint8)
        return quantized, arr_min, arr_max
    
    # Normal quantization
    quantized = ((arr - arr_min) / (arr_max - arr_min) * 255).round().astype(np.uint8)
    return quantized, arr_min, arr_max

def dequantize(quantized, arr_min, arr_max):

    if arr_max == arr_min:
        return np.full(quantized.shape, arr_min, dtype=np.float32)
    
    # Normal dequantization
    return quantized.astype(np.float32) / 255 * (arr_max - arr_min) + arr_min

def main():
    model = joblib.load("src/trained_model.joblib")
    coef = model.coef_
    intercept = np.atleast_1d(model.intercept_)

    joblib.dump({'coef_': coef, 'intercept_': intercept}, "src/unquant_params.joblib")

    # Calculate original model performance
    X_train, X_test, y_train, y_test = load_data()
    orig_preds = np.dot(X_test, coef) + intercept[0]
    orig_r2 = r2_score(y_test, orig_preds)
    orig_mse = mean_squared_error(y_test, orig_preds)

    # Quantize weights
    q_coef, coef_min, coef_max = min_max_quantize(coef)
    q_intercept, int_min, int_max = min_max_quantize(intercept)
    joblib.dump({
        'q_coef': q_coef,
        'coef_min': coef_min,
        'coef_max': coef_max,
        'q_intercept': q_intercept,
        'int_min': int_min,
        'int_max': int_max,
    }, "src/quant_params.joblib")

    dq_coef = dequantize(q_coef, coef_min, coef_max)
    dq_intercept = dequantize(q_intercept, int_min, int_max)[0]
    preds = np.dot(X_test, dq_coef) + dq_intercept
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    
    print(f"\n" + "="*50)
    print(f"RESULTS:")
    print(f"Original Model — R2: {orig_r2:.6f}, MSE: {orig_mse:.6f}")
    print(f"Quantized Model — R2: {r2:.4f}, MSE: {mse:.4f}")
    print(f"\n" + "="*50)

if __name__ == "__main__":
    main()