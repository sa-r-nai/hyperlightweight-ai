@echo off
python chat_native_200m.py --checkpoint "%~1" --device "%~2" --tokenizer "tokenizer/native_english_bpe.json"
