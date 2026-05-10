import tensorflow as tf
import numpy as np
import cv2
import os

# from keras.preprocessing import image

# ==============================
# LOAD MODEL
# ==============================

model = tf.keras.models.load_model("model.h5")

base_model = model.layers[0]

# ==============================
# FIND LAST CONV LAYER
# ==============================

last_conv_layer = None

for layer in reversed(base_model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer
        break

print("Last Conv Layer:", last_conv_layer.name)

# ==============================
# GRAD MODEL
# ==============================

grad_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=[last_conv_layer.output, base_model.output]
)

# ==============================
# TEST IMAGE FOLDER
# ==============================

folder_path = "dataset/test"

# ==============================
# OUTPUT FOLDER
# ==============================

os.makedirs("gradcam_outputs", exist_ok=True)

image_list = []

for subfolder in os.listdir(folder_path):
    sub_path = os.path.join(folder_path, subfolder)

    for img_name in os.listdir(sub_path):
        image_list.append(os.path.join(sub_path, img_name))

results = []

# ==============================
# PROCESS IMAGES
# ==============================

for img_path in image_list:

    img_name = os.path.basename(img_path)

    # ==============================
    # PREPROCESS IMAGE
    # ==============================

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ==============================
    # PREDICTION
    # ==============================

    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        label = "PNEUMONIA"
        confidence = pred
    else:
        label = "NORMAL"
        confidence = 1 - pred

    # ==============================
    # GRAD-CAM
    # ==============================

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    # ==============================
    # ORIGINAL IMAGE
    # ==============================

    original = cv2.imread(img_path)

    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # ==============================
    # OVERLAY
    # ==============================

    superimposed = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )

    output_path = os.path.join(
        "gradcam_outputs",
        img_name
    )

    cv2.imwrite(output_path, superimposed)

    # ==============================
    # SEVERITY SCORE
    # ==============================

    heatmap_intensity = np.mean(heatmap) / 255

    severity_score = (
        0.7 * confidence +
        0.3 * heatmap_intensity
    )

    # ==============================
    # PRIORITY LEVEL
    # ==============================

    if severity_score > 0.80:
        priority = "High 🚨"

    elif severity_score > 0.55:
        priority = "Medium"

    else:
        priority = "Low"

    results.append({
        "image": img_name,
        "severity": severity_score,
        "priority": priority,
        "confidence": confidence,
        "label": label
    })

# ==============================
# SORT RESULTS
# ==============================

results = sorted(
    results,
    key=lambda x: x["severity"],
    reverse=True
)

# ==============================
# PRINT OUTPUT
# ==============================

print("\n===== PATIENT PRIORITY LIST =====\n")

for i, r in enumerate(results):

    print(
        f"{i+1}. {r['image']} → "
        f"{r['priority']} "
        f"(Score: {round(r['severity'],2)}, "
        f"Conf: {round(r['confidence']*100,1)}%, "
        f"{r['label']})"
    )

print("\nGrad-CAM images saved in gradcam_outputs/")