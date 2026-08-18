class SentenceBuilder:

    def __init__(self):
        self.text = ""

    def add_letter(self, letter):
        """
        Add a confirmed letter to the current text.
        """
        if letter:
            self.text += letter

    def add_space(self):
        """
        Add a space between words.
        """
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def backspace(self):
        """
        Remove the last character.
        """
        self.text = self.text[:-1]

    def clear(self):
        """
        Clear the entire sentence.
        """
        self.text = ""

    def get_text(self):
        """
        Return the current text.
        """
        return self.text