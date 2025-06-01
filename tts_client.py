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
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return False

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

            if response.status_code == 200:
                print(f"Successfully uploaded voice sample: {file_path}")
                return True
            else:
                print(f"Error uploading voice sample: {response.status_code} - {response.text}")
                return False

    def generate_speech(self, text: str, voice_id: str = None, emotion_vector: List[float] = None,
                        language: str = "en-us", speaking_rate: float = None,
                        pitch: float = None, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate speech from text."""
        payload = {
            "text": text,
            "language": language
        }

        if voice_id:
            payload["voice_id"] = voice_id

        if emotion_vector:
            payload["emotion_vector"] = emotion_vector

        if speaking_rate:
            payload["speaking_rate"] = speaking_rate

        if pitch:
            payload["pitch"] = pitch

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

            # Upload voice samples for cloning if specified in voice info
            self._upload_voice_samples(voice_info)

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

    def _upload_voice_samples(self, voice_info: Dict[str, Any]):
        """Upload voice samples for cloning if specified."""
        voices = voice_info.get("voices", [])

        for voice in voices:
            voice_id = voice.get("id")
            name = voice.get("name", "Unknown")
            description = voice.get("description", "")

            cloning = voice.get("cloning", {})
            if cloning and cloning.get("enabled", False):
                audio_file = cloning.get("audio_file")
                if audio_file and os.path.exists(audio_file):
                    print(f"Uploading voice sample for {name} ({voice_id})...")
                    self.upload_voice_sample(voice_id, name, description, audio_file)
                else:
                    print(f"Warning: Voice sample file not found for {name}: {audio_file}")

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
        character = entry.get("voice", entry.get("character", ""))
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
        """Get the emotion vector for a given emotion name using the voice's parameters."""
        # Default emotion vector
        default_vector = [0.3077, 0.0256, 0.0256, 0.0256, 0.0256, 0.0256, 0.2564, 0.3077]

        if not voice:
            return default_vector

        # In the new format, we'll directly use the emotion_vector from zonos_parameters
        zonos_params = voice.get("zonos_parameters", {})
        if "emotion_vector" in zonos_params:
            return zonos_params["emotion_vector"]

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

            # Get emotional parameters
            emotion_vector = self._get_emotion_vector(emotion_name, voice)

            # Get voice-specific parameters
            voice_id = voice.get("id") if voice else None
            speaking_rate = None
            pitch = None

            if voice and "zonos_parameters" in voice:
                zonos_params = voice.get("zonos_parameters", {})
                speaking_rate = zonos_params.get("speaking_rate")
                pitch = zonos_params.get("pitch")

            # Generate a seed based on the resource name for consistency
            import hashlib
            seed = int(hashlib.md5(resource_name.encode()).hexdigest(), 16) % (2**31)

            result = self.generate_speech(
                text=text,
                voice_id=voice_id,
                emotion_vector=emotion_vector,
                speaking_rate=speaking_rate,
                pitch=pitch,
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

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Dialogue processing command
    dialogue_parser = subparsers.add_parser('dialogue', help='Process dialogue files')
    dialogue_parser.add_argument("--dialogue", type=str, required=True, help="Path to dialogue JSON file")
    dialogue_parser.add_argument("--voice-info", type=str, required=True, help="Path to voice info JSON file")
    dialogue_parser.add_argument("--output", type=str, default="output_audio", help="Output directory for audio files")

    # List voices command
    list_voices_parser = subparsers.add_parser('list-voices', help='List available voices')

    # Upload voice sample command
    upload_parser = subparsers.add_parser('upload-sample', help='Upload a voice sample')
    upload_parser.add_argument("--voice-id", type=str, required=True, help="Voice ID to upload sample for")
    upload_parser.add_argument("--name", type=str, required=True, help="Voice name")
    upload_parser.add_argument("--description", type=str, default="", help="Voice description")
    upload_parser.add_argument("--file", type=str, required=True, help="Path to audio file")

    # Upload voice info command
    upload_info_parser = subparsers.add_parser('upload-info', help='Upload voice info')
    upload_info_parser.add_argument("--voice-info", type=str, required=True, help="Path to voice info JSON file")

    # Generate speech command
    generate_parser = subparsers.add_parser('generate', help='Generate speech from text')
    generate_parser.add_argument("--text", type=str, required=True, help="Text to convert to speech")
    generate_parser.add_argument("--voice-id", type=str, help="Voice ID to use")
    generate_parser.add_argument("--output", type=str, default="output.wav", help="Output audio file")

    args = parser.parse_args()

    client = TTSClient(args.api)

    # Test connection
    if not client.test_connection():
        print("Failed to connect to TTS API server. Exiting.")
        sys.exit(1)

    # Process the command
    if args.command == 'list-voices':
        voices = client.list_voices()
        print("Available voices:")
        for voice in voices:
            print(f"  - {voice.get('id')}: {voice.get('name', '')}")
            cloning = voice.get("cloning", {})
            if cloning and cloning.get("enabled", False):
                print(f"    Voice cloning enabled: {cloning.get('audio_file', 'No sample file')}")

    elif args.command == 'upload-sample':
        success = client.upload_voice_sample(
            args.voice_id, args.name, args.description, args.file
        )
        if success:
            print(f"Successfully uploaded voice sample for {args.voice_id}")
        else:
            print(f"Failed to upload voice sample")

    elif args.command == 'upload-info':
        with open(args.voice_info, "r", encoding="utf-8") as f:
            voice_info = json.load(f)
        file_id = client.upload_voice_info(voice_info)
        print(f"Uploaded voice info with ID: {file_id}")

        # If voice info contains samples, upload those too
        client._upload_voice_samples(voice_info)

    elif args.command == 'generate':
        result = client.generate_speech(args.text, args.voice_id)
        if result:
            file_url = result.get("file_url")
            if file_url:
                success = client.download_file(file_url, args.output)
                if success:
                    print(f"Generated speech saved to {args.output}")
                else:
                    print("Failed to download generated speech")
            else:
                print("No file URL returned")
        else:
            print("Failed to generate speech")

    elif args.command == 'dialogue':
        client.process_dialogue_file(args.dialogue, args.voice_info, args.output)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()