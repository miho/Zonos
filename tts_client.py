# tts_client.py
import json
import os
import sys
import requests
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import tqdm

class TTSClient:
    def __init__(self, api_url: str):
        """Initialize the TTS client with the API URL."""
        self.api_url = api_url
        self.session = requests.Session()
        self.output_dir = Path("output_audio")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.voices = {}  # Cache for voice information

    def normalize_id(self, id_str):
        """Normalize ID for case-insensitive comparison."""
        if id_str is None:
            return ""
        return str(id_str).lower().strip()

    def test_connection(self) -> bool:
        """Test connection to the TTS API server."""
        try:
            response = self.session.get(f"{self.api_url}/")
            return response.status_code == 200
        except Exception as e:
            print(f"Error connecting to TTS API: {e}")
            return False

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the TTS system."""
        try:
            response = self.session.get(f"{self.api_url}/system-info")
            return response.json()
        except Exception as e:
            print(f"Error getting system info: {e}")
            return {}

    def list_voices(self) -> List[Dict[str, Any]]:
        """List all available voices on the server."""
        try:
            response = self.session.get(f"{self.api_url}/voices")
            data = response.json()
            return data.get("voices", [])
        except Exception as e:
            print(f"Error listing voices: {e}")
            return []

    def upload_voice_info(self, voice_info: Dict[str, Any]) -> str:
        """Upload voice information to the server."""
        response = self.session.post(
            f"{self.api_url}/voice-info",
            json={"voice_info": voice_info}
        )
        data = response.json()
        return data.get("file_id")

    def upload_voice_sample(self, voice_id: str, name: str, description: str, file_path: str) -> bool:
        """Upload a voice sample to the server."""
        with open(file_path, "rb") as f:
            files = {"sample_file": (os.path.basename(file_path), f)}
            data = {
                "voice_data": json.dumps({
                    "voice_id": voice_id,
                    "name": name,
                    "description": description
                })
            }
            response = self.session.post(f"{self.api_url}/voice-sample", files=files, data=data)
            return response.status_code == 200

    def generate_speech(self, text: str, voice_id: str = None, emotion: List[float] = None,
                        language: str = "en-us", speaking_rate: float = 15.0,
                        seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate speech from text."""
        payload = {
            "text": text,
            "language": language,
            "speaking_rate": speaking_rate
        }

        if voice_id:
            payload["voice_id"] = voice_id

        if emotion:
            payload["emotion"] = emotion

        if seed is not None:
            payload["seed"] = seed

        response = self.session.post(f"{self.api_url}/tts", json=payload)
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return None

        return response.json()

    def download_file(self, file_url: str, output_path: str) -> bool:
        """Download a file from the server."""
        try:
            response = self.session.get(f"{self.api_url}{file_url}", stream=True)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                print(f"Error downloading file: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False

    def process_dialogue_file(self, dialogue_file: str, voice_info_file: str,
                            output_dir: Optional[str] = None) -> bool:
        """Process a dialogue file and generate speech for each entry."""
        try:
            # Load the dialogue file
            with open(dialogue_file, "r", encoding="utf-8") as f:
                dialogue_data = json.load(f)

            # Load the voice info file
            with open(voice_info_file, "r", encoding="utf-8") as f:
                voice_info = json.load(f)

            # Upload voice info to server
            self.upload_voice_info(voice_info)

            # Set output directory
            if output_dir:
                self.output_dir = Path(output_dir)
                self.output_dir.mkdir(exist_ok=True, parents=True)

            # Extract voice mapping from voice info
            voice_mapping = self._extract_voice_mapping(voice_info)

            # Process each dialogue entry
            dialogues = dialogue_data.get("dialogues", [])
            print(f"Processing {len(dialogues)} dialogue entries...")

            for entry in tqdm.tqdm(dialogues):
                self._process_dialogue_entry(entry, voice_mapping)

            print(f"Completed processing dialogue file. Output saved to {self.output_dir}")
            return True

        except Exception as e:
            print(f"Error processing dialogue file: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _extract_voice_mapping(self, voice_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract voice mapping from voice info (case-insensitive)."""
        voices_by_id = {}

        # Extract voice definitions - Make case-insensitive
        for voice in voice_info.get("voices", []):
            voice_id = voice.get("id")
            if voice_id:
                # Store voice under lowercase key but preserve original voice
                voices_by_id[self.normalize_id(voice_id)] = voice

        # Extract voice mapping
        mapping = {}
        voice_mapping = voice_info.get("voice_mapping", {})

        # Process character defaults - Make case-insensitive
        for character, voice_id in voice_mapping.get("character_defaults", {}).items():
            if voice_id and self.normalize_id(voice_id) in voices_by_id:
                # Store character mapping with original case preserved
                mapping[character] = voices_by_id[self.normalize_id(voice_id)]

        # Process global defaults - Make case-insensitive
        for text_type, voice_id in voice_mapping.get("global_defaults", {}).items():
            if voice_id and self.normalize_id(voice_id) in voices_by_id:
                mapping[f"__default_{text_type}__"] = voices_by_id[self.normalize_id(voice_id)]

        return mapping

    def _get_voice_for_entry(self, entry: Dict[str, Any], voice_mapping: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Get the appropriate voice for a dialogue entry (case-insensitive)."""
        character = entry.get("character", "")
        text_type = entry.get("textType", "")

        # Try to find a direct match for the character (case-insensitive)
        for map_character, voice in voice_mapping.items():
            if self.normalize_id(map_character) == self.normalize_id(character):
                return voice

        # Try default for text type
        default_key = f"__default_{text_type}__"
        for map_key, voice in voice_mapping.items():
            if self.normalize_id(map_key) == self.normalize_id(default_key):
                return voice

        # Use narrator as fallback (case-insensitive)
        for map_character, voice in voice_mapping.items():
            if self.normalize_id(map_character) == "narrator":
                return voice

        # Return first voice as last resort
        if voice_mapping:
            return next(iter(voice_mapping.values()))

        # No voices available
        return None

    def _get_emotion_vector(self, emotion: str, voice: Dict[str, Any]) -> List[float]:
        """Get the emotion vector for a given emotion and voice."""
        # Default emotion vector
        default_vector = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]

        if not voice:
            return default_vector

        # Try to find the emotion in the voice's emotional profiles
        emotional_profiles = voice.get("emotional_profiles", {})

        # Case-insensitive emotion lookup
        for emotion_key, profile in emotional_profiles.items():
            if self.normalize_id(emotion_key) == self.normalize_id(emotion):
                if "emotion_vector" in profile:
                    return profile["emotion_vector"]

        # Return default vector if no matching emotion found
        return default_vector

    def _process_dialogue_entry(self, entry: Dict[str, Any], voice_mapping: Dict[str, Dict[str, Any]]) -> bool:
        """Process a single dialogue entry."""
        try:
            # Get the resource name and text
            resource_name = entry.get("resourceName")
            text = entry.get("text", "")

            if not text or not resource_name:
                return False

            # Output path
            output_path = self.output_dir / resource_name

            # If file already exists, skip
            if output_path.exists():
                print(f"Skipping existing file: {resource_name}")
                return True

            # Get voice and emotion
            voice = self._get_voice_for_entry(entry, voice_mapping)
            emotion_name = entry.get("emotion", "neutral")
            emotion_vector = self._get_emotion_vector(emotion_name, voice)

            # Generate speech
            voice_id = voice.get("id") if voice else None

            # Apply appropriate speaking rate
            speaking_rate = 15.0  # Default
            if voice and "default_parameters" in voice:
                speaking_rate = voice["default_parameters"].get("speaking_rate", 15.0)

            # Apply emotional speaking rate adjustment
            if voice and "emotional_profiles" in voice:
                # Case-insensitive emotion lookup
                for emotion_key, profile in voice["emotional_profiles"].items():
                    if self.normalize_id(emotion_key) == self.normalize_id(emotion_name):
                        speaking_rate_adjustment = profile.get("speaking_rate_adjustment", 0)
                        speaking_rate += speaking_rate_adjustment
                        break

            # Generate a seed based on the resource name for consistency
            import hashlib
            seed = int(hashlib.md5(resource_name.encode()).hexdigest(), 16) % (2**31)

            result = self.generate_speech(
                text=text,
                voice_id=voice_id,
                emotion=emotion_vector,
                speaking_rate=speaking_rate,
                seed=seed
            )

            if result:
                # Download the generated file
                file_url = result.get("file_url")
                if file_url:
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    return self.download_file(file_url, output_path)

            return False

        except Exception as e:
            print(f"Error processing dialogue entry: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="TTS Client for processing dialogue files")
    parser.add_argument("--api", type=str, default="http://localhost:8000", help="TTS API URL")

    # Create a group of arguments for when --list-voices is not specified
    dialogue_group = parser.add_argument_group('dialogue processing')
    dialogue_group.add_argument("--dialogue", type=str, help="Path to dialogue JSON file")
    dialogue_group.add_argument("--voice-info", type=str, help="Path to voice info JSON file")
    dialogue_group.add_argument("--output", type=str, default="output_audio", help="Output directory for audio files")

    # Add list-voices argument
    parser.add_argument("--list-voices", action="store_true", help="List available voices and exit")

    args = parser.parse_args()

    client = TTSClient(args.api)

    # Test connection
    if not client.test_connection():
        print("Failed to connect to TTS API server. Exiting.")
        sys.exit(1)

    # List voices if requested
    if args.list_voices:
        voices = client.list_voices()
        print("Available voices:")
        for voice in voices:
            print(f"  - {voice.get('id')}: {voice.get('name', '')}")
        sys.exit(0)

    # If not listing voices, ensure dialogue and voice-info are provided
    if not args.dialogue or not args.voice_info:
        parser.error("Both --dialogue and --voice-info are required when not using --list-voices")

    # Process dialogue file
    client.process_dialogue_file(args.dialogue, args.voice_info, args.output)

if __name__ == "__main__":
    main()