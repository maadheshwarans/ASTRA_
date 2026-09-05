import os
import numpy as np
import librosa
import tensorflow as tf

# ============================================================
# ASTRA-Edge — Category-Level False Activation Evaluation
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "astra_cnn.keras")
NORMALIZATION_PATH = os.path.join(PROJECT_DIR, "models", "normalization.npz")

SAMPLE_RATE = 16000
DURATION = 1.5
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)

N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160

THRESHOLD = 0.5

# Dataset categories
CATEGORIES = [
    "positive",
    "negative",
    "background",
    "hard_negatives"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 90)
print("     ASTRA-Edge — Category-Level False Activation Evaluation")
print("=" * 90)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded.")


# ============================================================
# LOAD NORMALIZATION
# ============================================================

print("\nLoading normalization...")

normalization = np.load(NORMALIZATION_PATH)

mean = normalization["mean"]
std = normalization["std"]

print("Normalization loaded.")

print("\nNormalization shapes:")
print(f"  mean: {mean.shape}")
print(f"  std : {std.shape}")


# ============================================================
# FIND SPEAKERS
# ============================================================

print("\nFinding speakers...")

speakers = []

for item in os.listdir(DATASET_DIR):

    item_path = os.path.join(DATASET_DIR, item)

    # Ignore files
    if not os.path.isdir(item_path):
        continue

    # Get directories inside this folder
    subdirectories = set()

    for name in os.listdir(item_path):

        path = os.path.join(item_path, name)

        if os.path.isdir(path):
            subdirectories.add(name.lower())

    # A valid speaker directory must contain at least
    # one of our dataset categories.
    if any(category in subdirectories for category in CATEGORIES):
        speakers.append(item)

speakers.sort()

print(f"Speakers found : {len(speakers)}")

for speaker in speakers:
    print(f"  {speaker}")


# ============================================================
# COLLECT AUDIO FILES
# ============================================================

audio_files = []

for speaker in speakers:

    speaker_path = os.path.join(DATASET_DIR, speaker)

    for category in CATEGORIES:

        category_path = os.path.join(speaker_path, category)

        if not os.path.isdir(category_path):
            continue

        for filename in sorted(os.listdir(category_path)):

            if not filename.lower().endswith(".wav"):
                continue

            filepath = os.path.join(category_path, filename)

            audio_files.append(
                {
                    "speaker": speaker,
                    "category": category,
                    "filename": filename,
                    "filepath": filepath
                }
            )


print(f"\nTotal files    : {len(audio_files)}")
print(f"Threshold      : prediction < {THRESHOLD} => ASTRA detected")


# ============================================================
# AUDIO → MFCC
# ============================================================

def extract_mfcc(filepath):

    try:

        # Load audio
        audio, sr = librosa.load(
            filepath,
            sr=SAMPLE_RATE,
            mono=True
        )

        # ----------------------------------------------------
        # Pad or truncate to exactly 1.5 seconds
        # ----------------------------------------------------

        if len(audio) < TARGET_SAMPLES:

            audio = np.pad(
                audio,
                (0, TARGET_SAMPLES - len(audio))
            )

        else:

            audio = audio[:TARGET_SAMPLES]

        # ----------------------------------------------------
        # MFCC
        # ----------------------------------------------------

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        # Add channel dimension
        mfcc = mfcc[..., np.newaxis]

        return mfcc

    except Exception as e:

        print(f"ERROR processing {filepath}")
        print(e)

        return None


# ============================================================
# NORMALIZE MFCC
# ============================================================

def normalize_mfcc(mfcc):

    """
    Normalize MFCC while preserving the expected
    model input shape:

        (13, 151, 1)
    """

    # Remove any accidental leading dimensions
    mfcc = np.asarray(mfcc, dtype=np.float32)

    while mfcc.ndim > 3:
        mfcc = np.squeeze(mfcc, axis=0)

    # --------------------------------------------------------
    # Per-MFCC normalization
    # mean/std shape:
    # (1, 13, 1, 1)
    # --------------------------------------------------------

    mean_values = np.squeeze(mean)
    std_values = np.squeeze(std)

    # mean_values/std_values should now be (13,)
    if mean_values.ndim == 1 and len(mean_values) == N_MFCC:

        mean_values = mean_values[:, np.newaxis, np.newaxis]
        std_values = std_values[:, np.newaxis, np.newaxis]

        mfcc = (
            mfcc - mean_values
        ) / (
            std_values + 1e-8
        )

    # --------------------------------------------------------
    # Fallback for scalar normalization
    # --------------------------------------------------------

    else:

        mean_value = float(np.mean(mean))
        std_value = float(np.mean(std))

        mfcc = (
            mfcc - mean_value
        ) / (
            std_value + 1e-8
        )

    return mfcc
# ============================================================
# RUN INFERENCE
# ============================================================

print("\nRunning inference...")

results = []

for item in audio_files:

    mfcc = extract_mfcc(item["filepath"])

    if mfcc is None:
        continue

    # Normalize
    mfcc = normalize_mfcc(mfcc)

    # Add batch dimension
    X = np.expand_dims(mfcc, axis=0)

    # Model prediction
    prediction = float(model.predict(X, verbose=0)[0][0])

    # IMPORTANT:
    #
    # Model output = probability of UNKNOWN
    #
    # Therefore:
    #
    # prediction < 0.5 → ASTRA
    # prediction >= 0.5 → UNKNOWN
    #
    astra_probability = 1.0 - prediction
    unknown_probability = prediction

    if prediction < THRESHOLD:
        predicted_class = "ASTRA"
    else:
        predicted_class = "UNKNOWN"

    results.append(
        {
            "speaker": item["speaker"],
            "category": item["category"],
            "filename": item["filename"],
            "astra_probability": astra_probability,
            "unknown_probability": unknown_probability,
            "prediction": predicted_class
        }
    )


# ============================================================
# DETAILED RESULTS
# ============================================================

print("\n" + "=" * 90)
print("DETAILED RESULTS")
print("=" * 90)

print(
    f"\n{'Speaker':<10}"
    f"{'Category':<18}"
    f"{'File':<32}"
    f"{'UNKNOWN':>10}"
    f"{'ASTRA':>10}"
    f"{'Prediction':>15}"
)

print("-" * 90)

for result in results:

    print(
        f"{result['speaker']:<10}"
        f"{result['category']:<18}"
        f"{result['filename']:<32}"
        f"{result['unknown_probability'] * 100:>9.2f}%"
        f"{result['astra_probability'] * 100:>9.2f}%"
        f"{result['prediction']:>15}"
    )


# ============================================================
# CATEGORY-LEVEL RESULTS
# ============================================================

print("\n" + "=" * 90)
print("PER-CATEGORY RESULTS")
print("=" * 90)


category_results = {}


for category in CATEGORIES:

    category_items = [
        r for r in results
        if r["category"] == category
    ]

    if len(category_items) == 0:
        continue

    total = len(category_items)

    astra_predictions = [
        r for r in category_items
        if r["prediction"] == "ASTRA"
    ]

    astra_count = len(astra_predictions)

    category_results[category] = {
        "total": total,
        "astra": astra_count
    }


# ------------------------------------------------------------
# POSITIVE
# ------------------------------------------------------------

if "positive" in category_results:

    total = category_results["positive"]["total"]
    detected = category_results["positive"]["astra"]

    detection_rate = detected / total * 100

    print("\nPOSITIVE (n={})".format(total))
    print("-" * 50)

    print(
        f"  True-positive rate   : {detection_rate:.2f}%"
    )

    print(
        f"  Correct detections   : {detected} / {total}"
    )

    print(
        f"  Missed detections    : {total - detected}"
    )


# ------------------------------------------------------------
# NEGATIVE
# ------------------------------------------------------------

if "negative" in category_results:

    total = category_results["negative"]["total"]
    false_activations = category_results["negative"]["astra"]

    false_activation_rate = (
        false_activations / total * 100
    )

    print("\nNEGATIVE (n={})".format(total))
    print("-" * 50)

    print(
        f"  False-activation rate: "
        f"{false_activation_rate:.2f}%"
    )

    print(
        f"  False activations    : "
        f"{false_activations} / {total}"
    )


# ------------------------------------------------------------
# BACKGROUND
# ------------------------------------------------------------

if "background" in category_results:

    total = category_results["background"]["total"]
    false_activations = category_results["background"]["astra"]

    false_activation_rate = (
        false_activations / total * 100
    )

    print("\nBACKGROUND (n={})".format(total))
    print("-" * 50)

    print(
        f"  False-activation rate: "
        f"{false_activation_rate:.2f}%"
    )

    print(
        f"  False activations    : "
        f"{false_activations} / {total}"
    )


# ------------------------------------------------------------
# HARD NEGATIVES
# ------------------------------------------------------------

if "hard_negatives" in category_results:

    total = category_results["hard_negatives"]["total"]
    false_activations = category_results["hard_negatives"]["astra"]

    false_activation_rate = (
        false_activations / total * 100
    )

    print("\nHARD_NEGATIVES (n={})".format(total))
    print("-" * 50)

    print(
        f"  False-activation rate: "
        f"{false_activation_rate:.2f}%"
    )

    print(
        f"  False activations    : "
        f"{false_activations} / {total}"
    )


# ============================================================
# PER-SPEAKER BREAKDOWN
# ============================================================

print("\n" + "=" * 90)
print("PER-SPEAKER BREAKDOWN")
print("=" * 90)

for speaker in speakers:

    speaker_results = [
        r for r in results
        if r["speaker"] == speaker
    ]

    if len(speaker_results) == 0:
        continue

    print(f"\nSpeaker: {speaker}")
    print("-" * 50)

    for category in CATEGORIES:

        category_items = [
            r for r in speaker_results
            if r["category"] == category
        ]

        if len(category_items) == 0:
            continue

        total = len(category_items)

        astra_count = sum(
            r["prediction"] == "ASTRA"
            for r in category_items
        )

        rate = astra_count / total * 100

        if category == "positive":

            print(
                f"  {category:<18}"
                f"{rate:>6.1f}% detected "
                f"(n={total})"
            )

        else:

            print(
                f"  {category:<18}"
                f"{rate:>6.1f}% false-activated "
                f"(n={total})"
            )


# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("OVERALL SUMMARY")
print("=" * 90)


# Positive detection rate
positive_results = [
    r for r in results
    if r["category"] == "positive"
]

positive_detected = sum(
    r["prediction"] == "ASTRA"
    for r in positive_results
)

if len(positive_results) > 0:

    overall_tpr = (
        positive_detected /
        len(positive_results) *
        100
    )

else:

    overall_tpr = 0.0


# Non-positive false activation rate
non_positive_results = [
    r for r in results
    if r["category"] != "positive"
]

false_activations = sum(
    r["prediction"] == "ASTRA"
    for r in non_positive_results
)

if len(non_positive_results) > 0:

    overall_far = (
        false_activations /
        len(non_positive_results) *
        100
    )

else:

    overall_far = 0.0


# Hard negative FAR
hard_negative_results = [
    r for r in results
    if r["category"] == "hard_negatives"
]

hard_negative_false_activations = sum(
    r["prediction"] == "ASTRA"
    for r in hard_negative_results
)

if len(hard_negative_results) > 0:

    hard_negative_far = (
        hard_negative_false_activations /
        len(hard_negative_results) *
        100
    )

else:

    hard_negative_far = 0.0


print(
    f"\nOverall true-positive rate (ASTRA)"
    f"      : {overall_tpr:.2f}%"
)

print(
    f"Overall false-activation rate (non-ASTRA)"
    f": {overall_far:.2f}%"
)

print(
    f"Hard-negative false-activation rate"
    f"      : {hard_negative_far:.2f}%"
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 90)
print("Evaluation complete.")
print("=" * 90)