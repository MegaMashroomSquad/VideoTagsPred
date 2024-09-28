import whisper
from moviepy.editor import VideoFileClip
import os
import random

def extract_audio_from_video(video_path, audio_path, start=None, end=None):
    """
    Extract audio from a video file and save it as a temporary WAV file.
    Optionally extract a specific segment of the video.
    """
    video = VideoFileClip(video_path)
    if start is not None and end is not None:
        video = video.subclip(start, end)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    video.close()

def transcribe_video(video_path, model_name="tiny", duration_limit=60, sampling_duration=30, num_samples=3):
    """
    Transcribe the audio track of a video file to text using Whisper.
    If the video duration exceeds the limit, it samples evenly distributed segments of the video.
    
    :param video_path: Path to the video file
    :param model_name: Whisper model to use (default: "tiny")
    :param duration_limit: Maximum duration to transcribe fully (in seconds)
    :param sampling_duration: Total duration of samples if video exceeds limit
    :return: Transcribed text
    """
    video = VideoFileClip(video_path)
    video_duration = video.duration
    video.close()

    model = whisper.load_model(model_name)
    temp_audio_path = "temp_audio.wav"

    if video_duration <= duration_limit:
        extract_audio_from_video(video_path, temp_audio_path)
        result = model.transcribe(temp_audio_path)
        os.remove(temp_audio_path)
        return result["text"]
    else:
        sample_duration = sampling_duration / num_samples
        transcriptions = []

        for i in range(num_samples):
            start_time = i * (video_duration - sample_duration) / (num_samples - 1)
            end_time = start_time + sample_duration
            extract_audio_from_video(video_path, temp_audio_path, start_time, end_time)
            result = model.transcribe(temp_audio_path)
            transcriptions.append(result["text"])
            os.remove(temp_audio_path)

        return " ".join(transcriptions)