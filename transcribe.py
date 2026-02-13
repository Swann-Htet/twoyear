"""
Transcribe rose_twoyear.mp3 with word-level timestamps using Whisper (stable-ts).
Outputs lyrics.json that the visualizer loads automatically.
"""

import stable_whisper
import json
import os
import sys

# Ensure ffmpeg binary is on PATH (bundled via imageio-ffmpeg)
try:
    import imageio_ffmpeg
    ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    # Also create a plain "ffmpeg.exe" symlink/copy name if needed
    ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dst = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_dst):
        import shutil
        shutil.copy2(ffmpeg_src, ffmpeg_dst)
    print(f"Using ffmpeg: {ffmpeg_dst}")
except ImportError:
    print("Warning: imageio-ffmpeg not found, hoping ffmpeg is on PATH")

AUDIO = os.path.join(os.path.dirname(__file__) or ".", "assets", "songs", "rose_twoyear.mp3")
OUT   = os.path.join(os.path.dirname(__file__), "assets", "lyrics.json")

print("Loading Whisper model (base)...")
model = stable_whisper.load_model("base")

print(f"Transcribing: {AUDIO}")
result = model.transcribe(AUDIO, word_timestamps=True)

# Stabilize timestamps for better accuracy
result = model.align(AUDIO, result)

# Build word-level data grouped by lines
words = []
line_idx = 0
prev_seg = -1

for segment in result.segments:
    for word_obj in segment.words:
        w = word_obj.word.strip()
        if not w:
            continue
        t = round(word_obj.start, 2)
        
        # Detect line breaks: new segment or long pause (>1.2s)
        if words and (segment.id != prev_seg):
            last_end = words[-1].get("_end", words[-1]["time"] + 0.3)
            if t - last_end > 0.8:
                line_idx += 1
        
        words.append({
            "word": w,
            "time": t,
            "line": line_idx,
            "_end": round(word_obj.end, 2)
        })
        prev_seg = segment.id

# Remove internal _end field
for w in words:
    del w["_end"]

# Build line text for reference
lines = {}
for w in words:
    lines.setdefault(w["line"], []).append(w["word"])
line_texts = {k: " ".join(v) for k, v in lines.items()}

output = {
    "words": words,
    "lines": line_texts,
    "totalWords": len(words),
    "totalLines": len(line_texts)
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✓ Done! {len(words)} words across {len(line_texts)} lines")
print(f"  Saved to: {OUT}")

# Preview first few lines
print("\nPreview:")
for i in range(min(6, len(line_texts))):
    first_word = next(w for w in words if w["line"] == i)
    print(f"  [{first_word['time']:6.2f}s] Line {i}: {line_texts[i]}")
