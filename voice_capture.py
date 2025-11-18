"""
Real-time Voice Capture Module
================================

This module captures audio from your microphone in real-time.

Features:
- Press-to-talk (press Enter to start, press Enter to stop)
- Automatic silence detection (stops after silence)
- Saves recording to a WAV file

Usage:
    recorder = VoiceRecorder()
    audio_file = recorder.record_auto_stop()   # Auto-stop on silence
    # or
    audio_file = recorder.record_press_to_talk()  # Manual control
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import logging
from typing import Optional
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceRecorder:
    """
    Records audio from microphone with auto-stop or manual control.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        silence_threshold=0.002,   # Lower = more sensitive to quiet ends
        silence_duration=2.5,      # Still allows natural pauses
        max_duration=60.0
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_duration = max_duration
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        logger.info("Voice Recorder initialized")


    def _calculate_rms(self, audio_data: np.ndarray) -> float:
        return np.sqrt(np.mean(audio_data**2))


    def record_auto_stop(self, output_file: str = "recorded_audio.wav") -> Optional[str]:
        print("\nPress ENTER to start recording...")
        input()  # Wait for first ENTER press to start
        
        print("🔴 Recording... (Press ENTER to stop)")
        self.recording = []
        
        # Start recording
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback
        ):
            self.is_recording = True
            input()  # Wait for second ENTER press to stop
            self.is_recording = False
            
        # Process the recorded audio
        if self.recording:
            audio_data = np.concatenate(self.recording, axis=0)
            sf.write(output_file, audio_data, self.sample_rate)
            duration = len(audio_data) / self.sample_rate
            print(f"\n✅ Saved: {output_file} ({duration:.1f}s)")
            return output_file
        else:
            print("\n❌ No audio recorded")
            return None

    def _audio_callback(self, indata, frames, time_info, status):
        """Simplified callback that just collects audio data"""
        if status:
            print(f"Audio status: {status}")
        if self.is_recording:
            self.recording.append(indata.copy())
    def record_press_to_talk(self, output_file: str = "recorded_audio.wav") -> Optional[str]:
        print("\n🎤 Press ENTER to START recording...")
        input()  # First press

        # Clear any leftover audio
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        print("🔴 Recording... Press ENTER to STOP")
        
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback
        )
        stream.start()
        
        try:
            input()  # Second press to stop
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted")
        finally:
            stream.stop()
            stream.close()

        # Collect all audio
        recorded_chunks = []
        while not self.audio_queue.empty():
            recorded_chunks.append(self.audio_queue.get())

        if recorded_chunks:
            audio_data = np.concatenate(recorded_chunks, axis=0)
            sf.write(output_file, audio_data, self.sample_rate)
            duration = len(audio_data) / self.sample_rate
            print(f"\n✅ Saved: {output_file} ({duration:.2f}s)\n")
            return output_file
        else:
            print("⚠️  No audio captured")
            return None

    def record_fixed_duration(
        self,
        duration: float,
        output_file: str = "recorded_audio.wav"
    ) -> str:
        print(f"\n🎤 Recording for {duration} seconds...")
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels
        )
        sd.wait()
        sf.write(output_file, audio_data, self.sample_rate)
        print(f"\n✅ Saved: {output_file}\n")
        return output_file


# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Voice Recorder")
    parser.add_argument('--mode', choices=['auto', 'fixed', 'press'], default='auto')
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--output', default='recorded_audio.wav')
    args = parser.parse_args()

    recorder = VoiceRecorder()

    if args.mode == 'auto':
        recorder.record_auto_stop(args.output)
    elif args.mode == 'fixed':
        recorder.record_fixed_duration(args.duration, args.output)
    elif args.mode == 'press':
        recorder.record_press_to_talk(args.output)