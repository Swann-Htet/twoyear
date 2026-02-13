"""
Force-align known lyrics of Rosé - "Two Years" to the audio
using stable-ts Whisper alignment. This gives exact word timestamps
since we provide the correct lyrics text.
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
    ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dst = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_dst):
        import shutil
        shutil.copy2(ffmpeg_src, ffmpeg_dst)
    print(f"Using ffmpeg: {ffmpeg_dst}")
except ImportError:
    print("Warning: imageio-ffmpeg not found, hoping ffmpeg is on PATH")

AUDIO = os.path.join(os.path.dirname(__file__) or ".", "assets", "songs", "rose_twoyear.mp3")
OUT   = os.path.join(os.path.dirname(__file__) or ".", "assets", "lyrics.json")

# ── Known correct lyrics with line groupings ──
LYRICS_LINES = [
    # Verse 1
    "How'd it all fall apart?",
    "You were right here before, in my arms",
    "Now you're invisible",
    "But the heartbreak's physical",
    "Got a place, moved away",
    "Somewhere with a different code, different state",
    "Still feels miserable",
    "God, it's so chemical",
    # Pre-Chorus
    "All that I know",
    "Is I can't let you go",
    # Chorus
    "It's been two years and you're still not gone",
    "Doesn't make sense that I can't move on",
    "Yeah, I try, I try, I try, I try",
    "But this love never dies",
    "Two years since you've been in my bed",
    "Even had a funeral for you in my head",
    "Yeah, I try, I try, I try, I try",
    "But this love never dies",
    # Verse 2
    "Another night, another vice",
    "Even try with someone new, someone nice",
    "I'll always hate the fact that you",
    "Ruined everybody after you",
    "I'm always coming back to you",
    # Chorus 2
    "It's been two years and you're still not gone",
    "Doesn't make sense that I can't move on",
    "Yeah, I try, I try, I try, I try",
    "But this love never dies",
    "Two years since you've been in my bed",
    "Even had a funeral for you in my head",
    "Yeah, I try, I try, I try, I try",
    "But this love never dies",
    
    "It's been two years and you're still not gone",
    "Doesn't make sense that I can't move on",
    "Yeah, I try, I try, I try, I try",
    "But this love never dies",
    "Two years since you've been in my bed",
    "Even had a funeral for you in my head",
    "Yeah, I try, I try, I try, I try",
    "But this love never dies",
]

# Join all lines into full text for alignment
FULL_TEXT = "\n".join(LYRICS_LINES)

print("Loading Whisper model (base)...")
model = stable_whisper.load_model("base")

print(f"Aligning known lyrics to: {AUDIO}")
# Force-align: we give it the correct text so it just finds WHEN each word is sung
result = model.align(AUDIO, FULL_TEXT, language="en")

# ── Build word-level data ──
# Extract all words from alignment result
all_aligned_words = []
for segment in result.segments:
    for word_obj in segment.words:
        w = word_obj.word.strip()
        if w:
            all_aligned_words.append({
                "word": w,
                "start": round(word_obj.start, 2),
                "end": round(word_obj.end, 2),
            })

print(f"Aligned {len(all_aligned_words)} words total")

# Map aligned words back to our line structure
words = []
idx = 0
for line_num, line_text in enumerate(LYRICS_LINES):
    expected_words = line_text.split()
    for wi, expected_word in enumerate(expected_words):
        if idx < len(all_aligned_words):
            aw = all_aligned_words[idx]
            words.append({
                "word": expected_word,
                "time": aw["start"],
                "line": line_num,
            })
            idx += 1
        else:
            last_t = words[-1]["time"] + 0.3 if words else 0
            words.append({
                "word": expected_word,
                "time": round(last_t, 2),
                "line": line_num,
            })

# Build line text reference
line_texts = {str(i): line for i, line in enumerate(LYRICS_LINES)}

# ── Post-process: fix out-of-order or jumped timestamps ──
print("\nPost-processing: fixing alignment anomalies...")

def fix_timestamps(words):
    """Fix words that jump out of order within or between lines."""
    fixed = 0
    
    for i in range(1, len(words)):
        prev_time = words[i-1]["time"]
        cur_time = words[i]["time"]
        
        # Same line: each word should be >= previous and within ~4s
        if words[i]["line"] == words[i-1]["line"]:
            if cur_time < prev_time or cur_time - prev_time > 4.0:
                # Interpolate: set to previous + reasonable gap
                words[i]["time"] = round(prev_time + 0.28, 2)
                fixed += 1
        else:
            # Different line: next line should be > previous line end, within ~8s
            # Allow bigger gap between lines, but not 20+ seconds
            if cur_time < prev_time:
                words[i]["time"] = round(prev_time + 0.6, 2)
                fixed += 1
            elif cur_time - prev_time > 10.0:
                # Huge gap — estimate based on average line duration
                # Get average time between line starts from good data
                words[i]["time"] = round(prev_time + 2.0, 2)
                fixed += 1
    
    return fixed

# Run multiple passes to stabilize
total_fixed = 0
for pass_num in range(5):
    n = fix_timestamps(words)
    total_fixed += n
    if n == 0:
        break

print(f"  Fixed {total_fixed} timestamps across {pass_num + 1} passes")

# Build section labels from line timing
section_map = {
    0: "Verse 1",
    8: "Pre-Chorus",
    10: "Chorus",
    18: "Verse 2",
    23: "Chorus",
}
sections = []
for line_num, label in sorted(section_map.items()):
    line_w = [w for w in words if w["line"] == line_num]
    if line_w:
        sections.append({"time": round(line_w[0]["time"] - 0.5, 2), "label": label})

output = {
    "words": words,
    "lines": line_texts,
    "sections": sections,
    "totalWords": len(words),
    "totalLines": len(line_texts),
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n=== Done! {len(words)} words across {len(line_texts)} lines ===")
print(f"  Saved to: {OUT}")

# Preview all lines with their start times
print("\nAll lines:")
for i, line in enumerate(LYRICS_LINES):
    line_w = [w for w in words if w["line"] == i]
    if line_w:
        t = line_w[0]["time"]
        print(f"  [{t:6.2f}s] L{i:02d}: {line}")
