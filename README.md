### Persian Swear Word Remover

This Python script removes Persian swear words from a given text file using a BERT model for sequence classification. It also provides additional information about the swear words detected in the text.

#### Features
- Load a pre-trained BERT model for Persian text classification.
- Remove swear words from the input text.
- Provide a cleaned version of the text.
- Output additional information such as the percentage of swear words and the list of detected swear words.

### Prerequisites
Before running the script, make sure you have the following installed:
- Python 3.6 or higher
- Required Python packages listed in `requirements.txt`

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/mohamadmahdifarokhi/Persian-Swear-Word.git
    cd Persian-Swear-Word
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the root directory with the following content:
    ```env
    MODEL_NAME=HooshvareLab/bert-fa-base-uncased-clf-persiannews
    SWEAR_FILE=swears.json
    ```

4. Ensure you have a JSON file (`swears.json`) with the following structure to define the Persian swear words:
    ```json
    {
        "swear_words": [
            "swear_word1",
            "swear_word2",
            "swear_word3",
            ...
        ]
    }
    ```

### Script Usage
The script takes an input text file, processes it to remove Persian swear words, and outputs the cleaned text along with additional information to an output text file.

#### Command-Line Arguments
- `text_file`: Path to the input text file to be cleaned.
- `output_file`: Path to the output text file for the cleaned text.

#### Example
1. Create an input text file (`input.txt`) with the following content:
    ```plaintext
    کیر و کص دالگت کسکش حرام زاده
    ```
    Replace "swear words" with actual Persian swear words from your `swears.json` list.

2. Run the script:
    ```bash
    python swear.py input.txt output.txt
    ```

3. The script will process the text and generate an `output.txt` file with the following content:
    ```plaintext
   Original Text:
   کیر و کص دالگت کسکش حرام زاده
   
   Cleaned Text:
   و دالگت
   
   Detected Swear Words:  کیر, کص, کسکش, حرام, زاده
   Percentage of Swear Words: 71.43%
    ```

### Code Explanation
The script consists of the following main components:

#### 1. Environment Configuration
Load environment variables from the `.env` file using `python-dotenv`.

```python
from dotenv import load_dotenv
import os

load_dotenv()
model_name = os.getenv('MODEL_NAME')
swear_file = os.getenv('SWEAR_FILE')
```

#### 2. Logging Configuration
Set up logging to track the script's progress and errors.

```python
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
```

#### 3. PersianSwearWordRemover Class
This class loads the BERT model and the list of swear words, normalizes the text, checks for swear words, and removes them.

```python
class PersianSwearWordRemover:
    def __init__(self, model_name, swear_file):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, return_all_scores=True)
        self.swear_words = self.load_swear_words(swear_file)

    @staticmethod
    def load_swear_words(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return data.get("swear_words", [])
        except Exception as e:
            logging.error(f"Error loading swear words from {file_path}: {e}")
            return []

    @staticmethod
    def normalize_text(text):
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def is_swear_word(self, token):
        normalized_token = self.normalize_text(token)
        return any(sw_word in normalized_token for sw_word in self.swear_words)

    def remove_persian_swear_words(self, text):
        normalized_text = self.normalize_text(text)
        tokens = normalized_text.split()
        cleaned_tokens = []
        detected_swear_words = []

        for token in tokens:
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
```

#### 4. Main Function
The `main()` function handles command-line arguments, reads the input file, processes the text, and writes the output to a file.

```python
def main():
    parser = argparse.ArgumentParser(description="Remove Persian swear words from text using BERT.")
    parser.add_argument('text_file', type=str, help='Path to the text file to be cleaned.')
    parser.add_argument('output_file', type=str, help='Path to the output text file for cleaned text.')
    args = parser.parse_args()

    try:
        with open(args.text_file, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        logging.error(f"Text file {args.text_file} not found.")
        return
    except Exception as e:
        logging.error(f"Error reading text file {args.text_file}: {e}")
        return

    remover = PersianSwearWordRemover(model_name=model_name, swear_file=swear_file)

    try:
        result = remover.remove_persian_swear_words(text)
    except Exception as e:
        logging.error(f"Error during text cleaning: {e}")
        return

    try:
        with open(args.output_file, 'w', encoding='utf-8') as file:
            file.write(f"Original Text:\n{text}\n\n")
            file.write(f"Cleaned Text:\n{result['cleaned_text']}\n\n")
            file.write(f"Detected Swear Words: {', '.join(result['detected_swear_words'])}\n")
            file.write(f"Percentage of Swear Words: {result['swear_word_percentage']:.2f}%\n")
        logging.info(f"Cleaned text and additional information written to {args.output_file}")
    except Exception as e:
        logging.error(f"Error writing to output file {args.output_file}: {e}")

if __name__ == "__main__":
    main()
```

### Summary
This script processes a text file to remove Persian swear words using a BERT model and outputs the cleaned text along with additional information about the swear words detected. It uses environment variables for configuration, logs progress and errors, and handles various exceptions gracefully. To run the script, follow the installation and usage instructions provided above.