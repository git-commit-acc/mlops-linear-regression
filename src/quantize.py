import joblib
import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
from utils import load_data
from sklearn.metrics import r2_score, mean_squared_error
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

def symmetric_quantize_int8(arr):
    qmin, qmax = -128, 127

    max_abs = np.max(np.abs(arr))
    if max_abs == 0:
        scale = 1.0
        quantized = np.zeros_like(arr, dtype=np.int8)
    else:
        scale = max_abs / qmax
        quantized = np.clip(np.round(arr / scale), qmin, qmax).astype(np.int8)

    return quantized, scale

def symmetric_dequantize_int8(quantized, scale):
    return quantized.astype(np.float32) * scale

def symmetric_quantize_int16(arr):
    qmin, qmax = -32768, 32767

    max_abs = np.max(np.abs(arr))
    if max_abs == 0:
        scale = 1.0
        quantized = np.zeros_like(arr, dtype=np.int16)
    else:
        scale = max_abs / qmax
        quantized = np.clip(np.round(arr / scale), qmin, qmax).astype(np.int16)

    return quantized, scale

def symmetric_dequantize_int16(quantized, scale):
    return quantized.astype(np.float32) * scale


def main():
    model = joblib.load("src/trained_model.joblib")
    coef = model.coef_
    intercept = np.atleast_1d(model.intercept_)

    joblib.dump({'coef_': coef, 'intercept_': intercept}, "src/unquant_params.joblib")

    X_train, X_test, y_train, y_test = load_data()

    # Evaluate original model
    orig_preds = np.dot(X_test, coef) + intercept[0]
    orig_r2 = r2_score(y_test, orig_preds)
    orig_mse = mean_squared_error(y_test, orig_preds)

    # Quantize model int8 parameters symmetrically
    q_coef_int8, coef_scale_int8 = symmetric_quantize_int8(coef)
    q_intercept_int8, int_scale_int8 = symmetric_quantize_int8(intercept)

    # Quantize model int16 parameters symmetrically
    q_coef_int16, coef_scale_int16 = symmetric_quantize_int16(coef)
    q_intercept_int16, int_scale_int16 = symmetric_quantize_int16(intercept)

    joblib.dump({
        'q_coef': q_coef_int8,
        'coef_scale': coef_scale_int8,
        'q_intercept': q_intercept_int8,
        'int_scale': int_scale_int8,
    }, "src/quant_params_int8.joblib")

    joblib.dump({
        'q_coef': q_coef_int16,
        'coef_scale': coef_scale_int16,
        'q_intercept': q_intercept_int16,
        'int_scale': int_scale_int16,
    }, "src/quant_params_int16.joblib")

    # Dequantize
    dq_coef_int8 = symmetric_dequantize_int8(q_coef_int8, coef_scale_int8)
    dq_intercept_int8 = symmetric_dequantize_int8(q_intercept_int8, int_scale_int8).item()

    dq_coef_int16 = symmetric_dequantize_int16(q_coef_int16, coef_scale_int16)
    dq_intercept_int16 = symmetric_dequantize_int16(q_intercept_int16, int_scale_int16).item()

    # Debug
    # print("\nSample coefficient comparison:")
    # print(f"Original coef[:5]:    {coef[:5]}")
    # print(f"Quantized 8 bit coef[:5]:   {q_coef_int8[:5]}")
    # print(f"Dequantized 8 bit coef[:5]: {dq_coef_int8[:5]}\n")
    # print(f"Quantized 16 bit coef[:5]:   {q_coef_int16[:5]}")
    # print(f"Dequantized 16 bit coef[:5]: {dq_coef_int16[:5]}\n")

    # Evaluate quantized model
    preds_int8 = np.dot(X_test, dq_coef_int8) + dq_intercept_int8
    r2_int8 = r2_score(y_test, preds_int8)
    mse_int8 = mean_squared_error(y_test, preds_int8)

    preds_int16 = np.dot(X_test, dq_coef_int16) + dq_intercept_int16
    r2_int16 = r2_score(y_test, preds_int16)
    mse_int16 = mean_squared_error(y_test, preds_int16)

    print(f"\n" + "="*50)
    print(f"RESULTS:")
    print(f"Original Model R2: {orig_r2:.4f}, MSE: {orig_mse:.4f}")
    print(f"Quantized Model 8 bit R2: {r2_int8:.4f}, MSE: {mse_int8:.4f}")
    print(f"Quantized Model 16 bit R2: {r2_int16:.4f}, MSE: {mse_int16:.4f}")
    print(f"\n" + "="*50)

if __name__ == "__main__":
    main()
