from runpod_handler import handler

mock_event = {
    "input": {
        "bucket": "your-bucket-name",
        "audio_key": "input/audio.wav",
        "image_key": "input/image.jpg"
    }
}

result = handler(mock_event)
print(result)
