
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Input, Model
from sklearn.metrics import average_precision_score

# Compute mAP
def compute_map(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return average_precision_score(y_true, y_pred)

# Build the model with early and late fusion
def build_fusion_model(video_dim=2048, audio_dim=128, hidden_dim=512):
    video_input = Input(shape=(video_dim,), name='video_input')
    audio_input = Input(shape=(audio_dim,), name='audio_input')

    # ========== Early Fusion ==========
    early_fusion = layers.Concatenate(name='early_fusion')([video_input, audio_input])
    early_fc = layers.Dense(hidden_dim, activation='softmax')(early_fusion)
    early_output = layers.Dense(1, activation='softmax', name='output_early')(early_fc)

    # ========== Mid Fusion (after pooling) ==========
    v_pool = layers.Reshape((video_dim, 1))(video_input)
    v_pool = layers.GlobalAveragePooling1D()(v_pool)
    a_pool = layers.Reshape((audio_dim, 1))(audio_input)
    a_pool = layers.GlobalAveragePooling1D()(a_pool)
    mid_fusion = layers.Concatenate(name='mid_fusion')([v_pool, a_pool])
    mid_fc = layers.Dense(hidden_dim, activation='softmax')(mid_fusion)
    mid_output = layers.Dense(1, activation='softmax', name='output_mid')(mid_fc)

    # ========== Late Fusion (after independent paths) ==========
    v_branch = layers.Dropout(0.6)(v_pool)
    a_branch = layers.Dropout(0.6)(a_pool)
    v_branch = layers.Dense(hidden_dim, activation='softmax')(v_branch)
    a_branch = layers.Dense(hidden_dim, activation='softmax')(a_branch)
    late_fusion = layers.Concatenate(name='late_fusion')([v_branch, a_branch])
    late_output = layers.Dense(1, activation='softmax', name='output_late')(late_fusion)

    model = Model(inputs=[video_input, audio_input],
                  outputs=[early_output, mid_output, late_output])
    return model

# Run simulation
def run_simulation():
    batch_size = 100
    video_dim = 2048
    audio_dim = 1024
    labels = np.random.randint(0, 2, size=(batch_size, 1)).astype(np.float32)

    video_data = np.random.randn(batch_size, video_dim).astype(np.float32)
    audio_data = np.random.randn(batch_size, audio_dim).astype(np.float32)

    model = build_fusion_model(video_dim, audio_dim)
    out_early, out_mid, out_late = model.predict([video_data, audio_data], verbose=0)

    map_early = compute_map(labels, out_early)
    map_mid = compute_map(labels, out_mid)
    map_late = compute_map(labels, out_late)

    print("\n📊 mAP Results for Fusion Types:")
    print(f"Early Fusion         → mAP: {map_early:.4f}")
    print(f"Mid-Level Fusion     → mAP: {map_mid:.4f}")
    print(f"Late Fusion          → mAP: {map_late:.4f}")

if __name__ == "__main__":
    run_simulation()
