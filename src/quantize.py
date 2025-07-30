import joblib
import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
from utils import load_data
from sklearn.metrics import r2_score, mean_squared_error

def q8_fn(vals, scale=None):
    """Quantize floats to uint8 with scaling."""
    if np.all(vals == 0):
        return np.zeros(vals.shape, dtype=np.uint8), 0.0, 0.0, 1.0
    if scale is None:
        absmax = np.abs(vals).max()
        scale = 250.0 / absmax if absmax > 0 else 1.0
    scaled = vals * scale
    mn, mx = scaled.min(), scaled.max()
    if mx == mn:
        quant = np.full(vals.shape, 127, dtype=np.uint8)
        return quant, mn, mx, scale
    rng = mx - mn
    norm = ((scaled - mn) / rng * 255)
    norm = np.clip(norm, 0, 255)
    quant = norm.astype(np.uint8)
    return quant, mn, mx, scale

def dq8_fn(quant, mn, mx, scale):
    """Dequantize uint8 to float using metadata."""
    rng = mx - mn
    if rng == 0:
        return np.full(quant.shape, mn / scale)
    scaled = (quant.astype(np.float32) / 255.0) * rng + mn
    return scaled / scale

def main():
    model = joblib.load("src/trained_model.joblib")
    coef = model.coef_
    intercept = np.atleast_1d(model.intercept_)

    # Save original parameters
    joblib.dump({'coef_': coef, 'intercept_': intercept}, "src/unquant_params.joblib")

    # Quantize using improved q8_fn
    q_coef, mn_c, mx_c, scale_c = q8_fn(coef)
    q_intercept, mn_i, mx_i, scale_i = q8_fn(intercept)

    # Save quantized weights
    joblib.dump({
        'q_coef': q_coef,
        'mn_c': mn_c,
        'mx_c': mx_c,
        'scale_c': scale_c,
        'q_intercept': q_intercept,
        'mn_i': mn_i,
        'mx_i': mx_i,
        'scale_i': scale_i,
    }, "src/quant_params.joblib")

    # Dequantize for inference
    dq_coef = dq8_fn(q_coef, mn_c, mx_c, scale_c)
    dq_intercept = dq8_fn(q_intercept, mn_i, mx_i, scale_i)[0]  # single float

    # Evaluate
    X_train, X_test, y_train, y_test = load_data()
    preds = np.dot(X_test, dq_coef) + dq_intercept

    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    print(f"Quantized Model — R2: {r2:.4f}, MSE: {mse:.4f}")

if __name__ == "__main__":
    main()