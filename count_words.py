# Word Count Program

def count_words(text):
    # Split the string by spaces
    words = text.split()
    return len(words)

# Take input from user
string = input("Enter a string: ")

# Count words
word_count = count_words(string)

print("Number of words in the given string:", word_count)




