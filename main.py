from download import download_vod, download_chat, render_chat
from chat_analysis import load_chat, chat_signals
from audio_analysis import analyze_speech_boundaries, audio_signals, extract_audio, refine_highlights_with_speech
from highlight_scoring import find_highlights
from clipper import cut_clips
from config import OUTPUT_DIR

def main():
    print(f"\nDownloading VOD...\n")
    download_vod()
    print(f"\nDownloading chat...\n")
    download_chat()
    print(f"\nRendering chat...\n")
    render_chat()

    print(f"\nExtracting audio...\n")
    extract_audio()

    chat = load_chat(f"{OUTPUT_DIR}/chat.json")
    if chat.empty:
        raise RuntimeError("Chat file loaded but contains no messages")

    chat_sig = chat_signals(chat)
    audio_sig = audio_signals(f"{OUTPUT_DIR}/audio.wav")

    print(f"\nFinding highlights...\n")
    highlights = find_highlights(chat_sig, audio_sig)
    
    print(f"\nAnalyzing speech boundaries with Whisper...\n")
    speech_boundaries = analyze_speech_boundaries(f"{OUTPUT_DIR}/audio.wav")
    
    print(f"\nRefining clip boundaries...\n")
    refined_highlights = refine_highlights_with_speech(highlights, speech_boundaries)
    
    print(f"\nCutting into clips...\n")
    cut_clips(f"{OUTPUT_DIR}/vod.mp4", refined_highlights, OUTPUT_DIR)

    print(f"\nGenerated {len(refined_highlights)} highlight clips")

if __name__ == "__main__":
    main()