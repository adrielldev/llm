import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")



text = "Akwirw ier" 
integers = tokenizer.encode(text)
strings = tokenizer.decode(integers)
print(strings)