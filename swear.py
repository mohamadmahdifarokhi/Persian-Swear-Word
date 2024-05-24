import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TextClassificationPipeline
import json
import argparse
import re
import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PersianSwearWordRemover:
    def __init__(self, model_name, swear_file):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, return_all_scores=True)
        self.swear_words = self.load_swear_words(swear_file)

    @staticmethod
    def load_swear_words(file_path):
        """
        Load Persian swear words from a JSON file.
        :param file_path: Path to the JSON file containing the swear words.
        :return: List of Persian swear words.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return data.get("swear_words", [])
        except Exception as e:
            logging.error(f"Error loading swear words from {file_path}: {e}")
            return []

    @staticmethod
    def normalize_text(text):
        """
        Normalize the input text by reducing repeated characters to their base form.
        :param text: The input text to be normalized.
        :return: Normalized text.
        """
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def is_swear_word(self, token):
        """
        Check if a token is a swear word by matching against the swear word list.
        :param token: The token to check.
        :return: Boolean indicating if the token is a swear word.
        """
        normalized_token = self.normalize_text(token)
        return any(sw_word in normalized_token for sw_word in self.swear_words)

    def remove_persian_swear_words(self, text):
        """
        Remove Persian swear words from the given text.
        :param text: The input text to be cleaned.
        :return: A dictionary containing the cleaned text, list of detected swear words, and percentage of swear words.
        """
        # Normalize the text to reduce repeated characters
        normalized_text = self.normalize_text(text)

        tokens = normalized_text.split()
        cleaned_tokens = []
        detected_swear_words = []

        for token in tokens:
            # Check if the token is classified as a swear word
            if self.is_swear_word(token):
                detected_swear_words.append(token)
            else:
                cleaned_tokens.append(token)

        cleaned_text = ' '.join(cleaned_tokens)
        swear_word_percentage = (len(detected_swear_words) / len(tokens)) * 100 if tokens else 0

        return {
            "cleaned_text": cleaned_text,
            "detected_swear_words": detected_swear_words,
            "swear_word_percentage": swear_word_percentage
        }


def main():
    # Set up argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Remove Persian swear words from text using BERT.")
    parser.add_argument('text_file', type=str, help='Path to the text file to be cleaned.')
    parser.add_argument('output_file', type=str, help='Path to the output text file for cleaned text.')
    args = parser.parse_args()

    # Read environment variables
    model_name = os.getenv('MODEL_NAME', 'HooshvareLab/bert-fa-base-uncased-clf-persiannews')
    swear_file = os.getenv('SWEAR_FILE', 'swears.json')

    try:
        # Read the input text from the file
        with open(args.text_file, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        logging.error(f"Text file {args.text_file} not found.")
        return
    except Exception as e:
        logging.error(f"Error reading text file {args.text_file}: {e}")
        return

    # Initialize the PersianSwearWordRemover
    remover = PersianSwearWordRemover(model_name=model_name, swear_file=swear_file)

    # Clean the input text
    try:
        result = remover.remove_persian_swear_words(text)
    except Exception as e:
        logging.error(f"Error during text cleaning: {e}")
        return

    # Write the cleaned text and additional information to the output file
    try:
        with open(args.output_file, 'w', encoding='utf-8') as file:
            file.write(f"Original Text:\n{text}\n\n")
            file.write(f"Cleaned Text:\n{result['cleaned_text']}\n\n")
            file.write(f"Detected Swear Words:\n{', '.join(result['detected_swear_words'])}\n")
            file.write(f"Percentage of Swear Words:\n{result['swear_word_percentage']:.2f}%\n")
        logging.info(f"Cleaned text and additional information written to {args.output_file}")
    except Exception as e:
        logging.error(f"Error writing to output file {args.output_file}: {e}")


if __name__ == "__main__":
    main()
