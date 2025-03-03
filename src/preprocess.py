import regex as re
import underthesea

from src.utils.Constant import TEEN_CODE_PATH

# Dictionary to store teen code mappings
teen_code_dict = {}

# Dictionary to map vowels to their respective indices
vowels_to_ids = {}

# Table to standardize accent typing for Vietnamese vowels
vowels_table = [
    ['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
    ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
    ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
    ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
    ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
    ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
    ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
    ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
    ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
    ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
    ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
    ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']
]

# Populate the vowels_to_ids dictionary with indices from vowels_table
for i in range(len(vowels_table)):
    for j in range(len(vowels_table[i]) - 1):
        vowels_to_ids[vowels_table[i][j]] = (i, j)

# Load teen_code dictionary when the module is imported
with open(TEEN_CODE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip().split('\t')
        if len(line) == 2:  # Ensure only valid lines are processed
            teen_code_dict[line[0]] = line[1]


# Standardize unicode
def convert_unicode(text: str) -> str:
    """
    Convert characters from Windows-1252 encoding to UTF-8 encoding.
    Args:
        text (str): The input text.
    Returns:
        str: The text with characters converted to UTF-8.
    """
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'
    charutf8 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'
    char1252 = char1252.split('|')
    charutf8 = charutf8.split('|')

    dic = {}
    for i in range(len(char1252)): dic[char1252[i]] = charutf8[i]
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dic[x.group()], text
    )


def is_valid_vietnamese_word(word: str) -> bool:
    """
     Check if a word is a valid Vietnamese word based on vowel positions.
     Args:
         word (str): The input word.
     Returns:
         bool: True if the word is valid, False otherwise.
     """
    chars = list(word)  # Convert the word into a list of characters
    vowel_indexes = -1  # Track the position of the last vowel found

    for index, char in enumerate(chars):
        x, y = vowels_to_ids.get(char, (-1, -1))  # Get vowel identifiers

        if x != -1:  # If the character is a vowel
            if vowel_indexes == -1:
                vowel_indexes = index  # Set the first vowel index
            else:
                if index - vowel_indexes != 1:  # Ensure vowels are consecutive
                    return False
                vowel_indexes = index  # Update the last vowel index

    return True  # Return True if all vowels are in a valid sequence


def standardize_word_typing(word: str) -> str:
    """
    Standardize the typing of a Vietnamese word by ensuring proper vowel placement
    and accent positioning according to Vietnamese orthographic rules.
    Args:
        word (str): The input word.
    Returns:
        str: The standardized word.
    """
    if not is_valid_vietnamese_word(word):
        return word  # Return the original word if it's not a valid Vietnamese word

    chars = list(word)  # Convert the word into a list of characters
    dau_cau = 0  # Store the accent mark (tone)
    vowel_indexes = []  # List to store the positions of vowels
    qu_or_gi = False  # Flag to handle special cases of 'qu' and 'gi'

    # Iterate through each character in the word
    for index, char in enumerate(chars):
        x, y = vowels_to_ids.get(char, (-1, -1))  # Get the vowel's position and tone ID

        if x == -1:  # If the character is not a vowel, continue
            continue
        elif x == 9:  # Check for "qu"
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'  # Ensure 'qu' is correctly represented
                qu_or_gi = True
        elif x == 5:  # Check for "gi"
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'  # Ensure 'gi' is correctly represented
                qu_or_gi = True

        if y != 0:  # If the character has a tone mark
            dau_cau = y  # Store the tone mark
            chars[index] = vowels_table[x][0]  # Replace with the unaccented base vowel

        # Only add vowels that are not part of "qu" or "gi" special cases
        if not qu_or_gi or index != 1:
            vowel_indexes.append(index)

    # Handle special cases for "qu" and "gi"
    if len(vowel_indexes) < 2:
        if qu_or_gi:
            if len(chars) == 2:  # If the word consists of only two characters
                x, y = vowels_to_ids.get(chars[1])
                chars[1] = vowels_table[x][dau_cau]  # Apply the tone mark
            else:
                x, y = vowels_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = vowels_table[x][dau_cau]
                else:
                    # Apply the tone to 'i' or 'u' accordingly
                    chars[1] = vowels_table[5][dau_cau] if chars[1] == 'i' else vowels_table[9][dau_cau]
            return ''.join(chars)  # Return the corrected word
        return word  # Return the original word if it does not meet conditions

    # Prioritize applying the tone mark on "ê" or "ơ" if present
    for index in vowel_indexes:
        x, y = vowels_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = vowels_table[x][dau_cau]
            return ''.join(chars)

    # If there are exactly two vowels
    if len(vowel_indexes) == 2:
        if vowel_indexes[-1] == len(chars) - 1:  # If the last vowel is at the end
            x, y = vowels_to_ids[chars[vowel_indexes[0]]]
            chars[vowel_indexes[0]] = vowels_table[x][dau_cau]  # Apply tone to first vowel
        else:
            x, y = vowels_to_ids[chars[vowel_indexes[1]]]
            chars[vowel_indexes[1]] = vowels_table[x][dau_cau]  # Apply tone to second vowel
    else:
        # If more than two vowels, apply the tone to the second vowel
        x, y = vowels_to_ids[chars[vowel_indexes[1]]]
        chars[vowel_indexes[1]] = vowels_table[x][dau_cau]

    return ''.join(chars)  # Return the standardized word


def standardize_sentence_typing(text: str) -> str:
    """
    Standardize the typing of a Vietnamese sentence.
    Args:
        text (str): The input sentence.
    Returns:
        str: The standardized sentence.
    """
    words = text.lower().split()
    for index, word in enumerate(words):
        # Separate leading and trailing punctuation from the word using regex
        # This regex attempts to extract:
        #   - Prefix punctuation (^...): e.g., "..." in "...hello"
        #   - Core word ([p{L}.]*\p{L}+): the actual Vietnamese word
        #   - Suffix punctuation (...$): e.g., "!!!" in "hello!!!"
        # However, '\p{L}' and '\p{P}' are Unicode properties that Python’s re module does not support.
        # We need to replace them with '\w' (word characters) or use the 'regex' library.
        cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')

        # If the word contains leading or trailing punctuation, apply standardization only to the core word
        if len(cw) == 3:
            cw[1] = standardize_word_typing(cw[1])  # Standardize the Vietnamese word

        # Reassemble the word, keeping punctuation intact
        words[index] = ''.join(cw)

    # Join words back into a sentence and return
    return ' '.join(words)


# Normalize acronyms
def normalize_acronyms(text: str) -> str:
    """
        Normalize acronyms in the text using the teen code dictionary.
        Args:
            text (str): The input text.
        Returns:
            str: The text with acronyms normalized.
        """
    words = []
    for word in text.strip().split():
        if word not in teen_code_dict.keys():
            words.append(word)
        else:
            words.append(teen_code_dict[word])
    return ' '.join(words)

def remove_unnecessary_characters(text: str) -> str:
    """
      Remove unnecessary characters from the text.
      Args:
          text (str): The input text.
      Returns:
          str
    """
    text = re.sub(
        r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÍÌỈĨỊÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ_]',
        ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text


def preprocess_fn(text: str) -> str:
    """
    Preprocess the input text by performing various normalization and cleaning steps.
    Args:
        text (str): The input text.
    Returns:
        str: The preprocessed text.
    """
    text = text.strip().lower()  # Normalize to lowercase & remove extra whitespace
    text = convert_unicode(text)  # Convert Unicode characters to standard form
    text = normalize_acronyms(text)  # Replace acronyms (teen code)
    text = standardize_sentence_typing(text)  # Standardize Vietnamese typing
    text = remove_unnecessary_characters(text)  # Remove unnecessary characters
    text = underthesea.word_tokenize(text, format="text")  # Tokenize words
    return text
