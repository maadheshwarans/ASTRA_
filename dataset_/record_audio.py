import os
import time
import random

import sounddevice as sd
from scipy.io.wavfile import write


# ============================================================
# ASTRA-Edge Dataset Recorder
# ============================================================

SAMPLE_RATE = 16000
DURATION = 1.5
CHANNELS = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Recording plan
# ============================================================

RECORDING_PLAN = {
    "positive": 60,
    "negative": 40,
    "hard_negatives": 20,
    "background": 10
}


# ============================================================
# ASTRA recording variations
# ============================================================

ASTRA_INSTRUCTIONS = [
    "Say ASTRA normally.",
    "Say ASTRA slightly louder.",
    "Say ASTRA slightly quieter.",
    "Say ASTRA at a faster speed.",
    "Say ASTRA at a slower speed.",
    "Say ASTRA naturally.",
    "Say ASTRA while sitting normally.",
    "Say ASTRA from about 30 cm away.",
    "Say ASTRA from about 1 meter away.",
    "Say ASTRA from about 2 meters away.",
    "Say ASTRA with the microphone slightly to your side.",
    "Say ASTRA with a different natural voice variation."
]


# ============================================================
# Negative examples
# ============================================================

NEGATIVE_INSTRUCTIONS = [
    "Say a normal unrelated word.",
    "Say a short unrelated word.",
    "Say a random everyday word.",
    "Say a different word naturally."
]


# ============================================================
# Hard negative examples
# ============================================================

HARD_NEGATIVE_INSTRUCTIONS = [
    "Say ASTRO.",
    "Say ASTER.",
    "Say EXTRA.",
    "Say ESTRA.",
    "Say ASTRAA.",
    "Say a word that sounds somewhat similar to ASTRA.",
    "Say a similar-sounding word naturally.",
    "Say a word that could easily be confused with ASTRA."
]


# ============================================================
# Utility functions
# ============================================================

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def get_next_number(folder_path, category):

    if not os.path.exists(folder_path):
        return 1

    numbers = []

    prefix = category + "_"

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(".wav"):
            continue

        if not filename.startswith(prefix):
            continue

        number_part = filename[len(prefix):-4]

        try:
            numbers.append(int(number_part))
        except ValueError:
            pass

    if not numbers:
        return 1

    return max(numbers) + 1


def count_recordings(folder_path):

    if not os.path.exists(folder_path):
        return 0

    return sum(
        1
        for filename in os.listdir(folder_path)
        if filename.lower().endswith(".wav")
    )


def countdown():

    print()
    print("Get ready...")

    for number in [3, 2, 1]:

        print(number)
        time.sleep(1)

    print()
    print(">>> RECORDING <<<")


def record_audio(device_number):

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        device=device_number
    )

    sd.wait()

    return audio


def get_input_device():

    print()
    print("=" * 70)
    print("                     MICROPHONE SETUP")
    print("=" * 70)
    print()

    devices = sd.query_devices()

    input_devices = []

    for index, device in enumerate(devices):

        if device["max_input_channels"] > 0:

            input_devices.append(index)

            print(
                f"[{index}] {device['name']}"
            )

    print()

    while True:

        try:

            device_number = int(
                input("Enter microphone device number: ")
            )

            if device_number in input_devices:
                return device_number

            print("That is not a valid input device.")

        except ValueError:

            print("Please enter a number.")


def get_speaker_id():

    print()
    print("=" * 70)
    print("                      SPEAKER SETUP")
    print("=" * 70)
    print()

    print("Use IDs such as:")
    print("1")
    print("2")
    print("3")
    print("4")
    print()

    while True:

        speaker_id = input(
            "Enter speaker ID: "
        ).strip()

        if speaker_id:
            return speaker_id

        print("Speaker ID cannot be empty.")


# ============================================================
# Display progress
# ============================================================

def show_progress(
    speaker_id,
    category,
    current,
    target,
    total_completed,
    total_target
):

    clear_screen()

    print("=" * 70)
    print("                 ASTRA-Edge DATASET RECORDER")
    print("=" * 70)
    print()

    print(f"Speaker       : {speaker_id}")
    print(f"Category      : {category}")
    print(f"Progress      : {current} / {target}")
    print(
        f"Overall       : "
        f"{total_completed} / {total_target}"
    )

    print()
    print("-" * 70)
    print()

    percentage = (
        current / target * 100
        if target > 0
        else 0
    )

    print(
        f"Category progress: {percentage:.1f}%"
    )

    print()


# ============================================================
# Record one category
# ============================================================

def record_category(
    speaker_id,
    category,
    target,
    device_number,
    total_completed,
    total_target
):

    folder_path = os.path.join(
        BASE_DIR,
        speaker_id,
        category
    )

    os.makedirs(
        folder_path,
        exist_ok=True
    )

    existing = count_recordings(
        folder_path
    )

    if existing >= target:

        print(
            f"{category}: already complete "
            f"({existing}/{target})"
        )

        return existing - target + total_completed

    remaining = target - existing

    starting_number = get_next_number(
        folder_path,
        category
    )

    for i in range(remaining):

        current = existing + i + 1

        show_progress(
            speaker_id,
            category,
            current,
            target,
            total_completed + i,
            total_target
        )

        # ----------------------------------------------------
        # Choose instruction
        # ----------------------------------------------------

        if category == "positive":

            instruction = random.choice(
                ASTRA_INSTRUCTIONS
            )

        elif category == "negative":

            instruction = random.choice(
                NEGATIVE_INSTRUCTIONS
            )

        elif category == "hard_negatives":

            instruction = random.choice(
                HARD_NEGATIVE_INSTRUCTIONS
            )

        else:

            instruction = (
                "Record the surrounding environment "
                "or remain silent."
            )

        print()
        print("Instruction:")
        print()
        print(f"    {instruction}")
        print()

        input(
            "Press ENTER when ready..."
        )

        countdown()

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        audio = record_audio(
            device_number
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        recording_number = (
            starting_number + i
        )

        filename = (
            f"{category}_"
            f"{recording_number:03d}.wav"
        )

        filepath = os.path.join(
            folder_path,
            filename
        )

        write(
            filepath,
            SAMPLE_RATE,
            audio
        )

        print()
        print("Saved:")
        print(filepath)

        # ----------------------------------------------------
        # Small pause
        # ----------------------------------------------------

        if i < remaining - 1:

            print()
            print(
                "Next recording in 2 seconds..."
            )

            time.sleep(2)

    return total_completed + remaining


# ============================================================
# Dataset overview
# ============================================================

def show_dataset_status(speaker_id):

    print()
    print("=" * 70)
    print("                    DATASET STATUS")
    print("=" * 70)
    print()

    speaker_dir = os.path.join(
        BASE_DIR,
        speaker_id
    )

    total = 0

    for category, target in RECORDING_PLAN.items():

        folder = os.path.join(
            speaker_dir,
            category
        )

        count = count_recordings(
            folder
        )

        total += count

        status = (
            "COMPLETE"
            if count >= target
            else "INCOMPLETE"
        )

        print(
            f"{category:<18}"
            f"{count:>3} / {target:<3}"
            f"   {status}"
        )

    print()
    print(
        f"Total recordings: {total}"
    )

    print()


# ============================================================
# Main
# ============================================================

def main():

    clear_screen()

    print("=" * 70)
    print("                 ASTRA-Edge DATASET RECORDER")
    print("=" * 70)
    print()

    print("Recording configuration:")
    print()
    print(f"Sample rate : {SAMPLE_RATE} Hz")
    print(f"Duration    : {DURATION} seconds")
    print("Channels    : Mono")
    print()

    print("Target per speaker:")
    print()

    for category, count in RECORDING_PLAN.items():

        print(
            f"  {category:<18}: {count}"
        )

    total_target = sum(
        RECORDING_PLAN.values()
    )

    print()
    print(
        f"Total per speaker: {total_target}"
    )

    # --------------------------------------------------------
    # Microphone
    # --------------------------------------------------------

    device_number = get_input_device()

    # --------------------------------------------------------
    # Speaker
    # --------------------------------------------------------

    speaker_id = get_speaker_id()

    # --------------------------------------------------------
    # Existing dataset
    # --------------------------------------------------------

    show_dataset_status(
        speaker_id
    )

    print(
        "Existing recordings will NOT be overwritten."
    )

    print()

    input(
        "Press ENTER to begin the recording session..."
    )

    # --------------------------------------------------------
    # Count existing recordings
    # --------------------------------------------------------

    speaker_dir = os.path.join(
        BASE_DIR,
        speaker_id
    )

    total_completed = 0

    for category in RECORDING_PLAN:

        folder = os.path.join(
            speaker_dir,
            category
        )

        total_completed += count_recordings(
            folder
        )

    # --------------------------------------------------------
    # Record categories
    # --------------------------------------------------------

    for category, target in RECORDING_PLAN.items():

        total_completed = record_category(
            speaker_id,
            category,
            target,
            device_number,
            total_completed,
            total_target
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    clear_screen()

    print("=" * 70)
    print("                 RECORDING SESSION COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Speaker: {speaker_id}"
    )

    print()

    show_dataset_status(
        speaker_id
    )

    print("=" * 70)
    print()
    print("ASTRA-Edge dataset recording complete.")
    print()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print()
        print("=" * 70)
        print("Recording stopped by user.")
        print("Already saved recordings are safe.")
        print("=" * 70)
        print()

    except Exception as error:

        print()
        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print()
        print(error)
        print()