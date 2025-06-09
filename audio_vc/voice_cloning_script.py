import csv
import json
from pathlib import Path
import outetts


def read_text_from_csv(file_path):
    texts = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            texts.extend(row)
    return texts


def read_text_from_json(file_path):
    with open(file_path, encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
        if isinstance(data, list):
            return [str(item) for item in data]
        elif isinstance(data, dict):
            return [str(value) for value in data.values()]
        else:
            return [str(data)]


def read_text_from_file(file_path):
    ext = Path(file_path).suffix.lower()
    if ext == '.csv':
        return read_text_from_csv(file_path)
    elif ext == '.json':
        return read_text_from_json(file_path)
    else:
        raise ValueError("Unsupported file type. Only CSV and JSON files are supported.")


def convert_text_to_speech(voice_path, text, suffix):
    interface = outetts.Interface(
        config=outetts.ModelConfig.auto_config(
            model=outetts.Models.VERSION_1_0_SIZE_1B,
            backend=outetts.Backend.LLAMACPP,
            quantization=outetts.LlamaCppQuantization.FP16
        )
    )

    speaker = interface.create_speaker(voice_path)
    interface.save_speaker(speaker, "speaker.json")
    speaker = interface.load_speaker("speaker.json")
    
    output = interface.generate(
        config=outetts.GenerationConfig(
            text,
            generation_type=outetts.GenerationType.CHUNKED,
            speaker=speaker,
            sampler_config=outetts.SamplerConfig(
                temperature=0.4
            ),
        )
    )
    output.save(f"vc_output_{suffix}.wav")


def main():
    file_path = input("Enter the path to the CSV or JSON file: ").strip()
    texts = read_text_from_file(file_path)
    
    voice_path = input("Enter the path to the voice example file: ").strip()

    for i in range(len(texts)):
        text = texts[i]
        print(text)
        convert_text_to_speech(voice_path=voice_path, text=text, suffix=i+1)


if __name__ == "__main__":
    main()
