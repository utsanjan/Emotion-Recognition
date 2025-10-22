#!/usr/bin/env python3
"""
Generate a short MIDI melody based on an emotion label.
Each emotion is mapped to a distinct musical scale, tempo, and note duration.
"""
import pretty_midi
import numpy as np
import argparse
import os

emotion_map = {
    'Angry': {'scale': [48, 51, 53, 55, 58, 60], 'tempo': 140, 'dur': 0.25},
    'Disgust': {'scale': [45, 48, 50, 53, 55], 'tempo': 80, 'dur': 0.5},
    'Fear': {'scale': [50, 53, 55, 58, 60, 63], 'tempo': 100, 'dur': 0.4},
    'Happy': {'scale': [60, 62, 64, 65, 67, 69, 71], 'tempo': 120, 'dur': 0.5},
    'Sad': {'scale': [48, 50, 52, 53, 55, 57], 'tempo': 70, 'dur': 0.6},
    'Surprise': {'scale': [60, 64, 67, 72], 'tempo': 140, 'dur': 0.25},
    'Neutral': {'scale': [60, 62, 64, 65, 67], 'tempo': 90, 'dur': 0.5}
}


def generate_melody(emotion, length=16, out_path='outputs/generated_music/melody.mid'):
    """Generate and save a MIDI melody for the given emotion."""
    if emotion not in emotion_map:
        print(f"Emotion '{emotion}' not found, using Neutral instead.")
        emotion = 'Neutral'

    info = emotion_map[emotion]
    scale = info['scale']
    tempo = info['tempo']
    dur = info['dur']

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    start = 0.0
    for _ in range(length):
        note_pitch = int(np.random.choice(scale))
        note_duration = dur + np.random.uniform(-0.05, 0.05)
        note = pretty_midi.Note(
            velocity=100,
            pitch=note_pitch,
            start=start,
            end=start + note_duration
        )
        instrument.notes.append(note)
        start += note_duration

    pm.instruments.append(instrument)
    pm.write(out_path)
    print(f"MIDI file saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--emotion', type=str, required=True, help='Emotion name (Happy, Sad, Angry, etc.)')
    parser.add_argument('--length', type=int, default=16, help='Number of notes to generate')
    parser.add_argument('--out', type=str, default='outputs/generated_music/melody.mid', help='Output MIDI path')
    args = parser.parse_args()

    generate_melody(args.emotion, length=args.length, out_path=args.out)