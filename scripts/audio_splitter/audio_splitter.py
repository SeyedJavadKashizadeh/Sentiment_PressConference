import json
import os
from pathlib import Path
from pydub import AudioSegment

# ====== CONFIG ======
base_dir = Path(r"your_directory")
json_dir = base_dir / "timestamped_text"
audio_dir = base_dir / "audio_files"
output_root = base_dir / "audio_files_split"
# ====================

output_root.mkdir(parents=True, exist_ok=True)

def is_number(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False

for json_path in json_dir.glob("*.json"):
    session_name = json_path.stem
    audio_path = audio_dir / f"{session_name}.wav"

    if not audio_path.exists():
        print(f" No matching audio file for {session_name}, skipping.")
        continue

    output_dir = output_root / session_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n Processing session: {session_name}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audio = AudioSegment.from_file(audio_path)

    for block in data:
        speaker = block.get("speaker", "")
        if "CHAIR" not in speaker.upper():
            continue

        for sent in block.get("sentences", []):
            start = sent.get("start_at")
            end = sent.get("end_at")

            if not is_number(start) or not is_number(end):
                continue

            start_ms, end_ms = float(start) * 1000, float(end) * 1000
            if end_ms <= start_ms:
                continue

            filename = f"{speaker.replace(' ', '_')}_s{sent['s_id']}.wav"
            output_path = output_dir / filename

            # Skip if already exists
            if output_path.exists():
                print(f" Already exists: {filename}")
                continue

            segment = audio[start_ms:end_ms]
            segment.export(output_path, format="wav")
            print(f" Saved: {filename}")

    print(f"Finished: {session_name}")

print("\n All sessions processed. Output in:", output_root)
