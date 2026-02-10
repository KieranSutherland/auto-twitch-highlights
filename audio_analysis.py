from config import OUTPUT_DIR
import ffmpeg
import whisper
import librosa

def extract_audio():
    (
        ffmpeg
        .input(f"{OUTPUT_DIR}/vod.mp4")
        .output(f"{OUTPUT_DIR}/audio.wav", vn=None)
        .overwrite_output()
        .run(quiet=True)
    )

def audio_signals(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    hop = sr

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_norm = (rms - rms.mean()) / rms.std()

    return rms_norm.tolist()

def analyze_speech_boundaries(audio_path):
    """
    Use Whisper to transcribe audio and detect natural speaking boundaries.
    Returns timestamps where topics/sentences end.
    """
    print(f"\nLoading Whisper model...\n")
    model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
    
    print(f"\nTranscribing audio...\n")
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        verbose=False,
        fp16=False  # Use FP32 on CPU
    )
    
    # Extract segment boundaries (natural pauses between sentences/topics)
    boundaries = []
    for segment in result['segments']:
        boundaries.append({
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'],
            'no_speech_prob': segment.get('no_speech_prob', 0)
        })
    
    return boundaries

def refine_highlights_with_speech(highlights, speech_boundaries, max_extension=5.0):
    """
    Extend highlight endpoints to align with natural speech boundaries.
    
    Args:
        highlights: List of (start, end) tuples in seconds
        speech_boundaries: List of segment dicts from Whisper
        max_extension: Maximum seconds to extend a clip to find boundary
    """
    refined = []
    
    for start, end in highlights:
        # Find the nearest speech boundary after the initial end point
        best_end = end
        min_distance = float('inf')
        
        for segment in speech_boundaries:
            seg_end = segment['end']
            
            # Only consider boundaries after our current end point
            # and within max_extension seconds
            if seg_end >= end and seg_end <= end + max_extension:
                distance = seg_end - end
                
                # Prefer boundaries with lower no_speech_prob (more confident speech end)
                # and closer to original end point
                score = distance + (segment['no_speech_prob'] * 2)
                
                if score < min_distance:
                    min_distance = score
                    best_end = seg_end
        
        refined.append((start, best_end))
    
    return refined