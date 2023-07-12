import re


def format_list_items(text):
    """Detects and formats all list items in a text.

    Args:
      text: The text to be processed.

    Returns:
       Formatted text of list items in markdown.
    """

    list_items = re.findall(
        r'(\d+\.\s[a-zA-Z0-9\s]+|\*\s[a-zA-Z0-9\s]+|\-\s[a-zA-Z0-9\s]+)', text)
    for item in list_items:
        text = text.replace(item, f"\n {item}")
    return text


if __name__ == "__main__":
    text = "To add custom fonts in Flutter, follow these steps: 1. Create a folder called fonts in the project’s root directory. 2. Add your.ttf,.otf, or.ttc font file into the fonts folder. 3. Open the pubspec.yaml file within the project and find the flutter section. 4. Add your custom font(s) under the fonts section. 5. Use the font file in your UI components as needed."

    print(format_list_items(text))
