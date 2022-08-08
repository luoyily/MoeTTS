from vitstext.symbols import symbols


def custom_cleaners(text):
  '''Only remove char not in symbols'''
  text = ''.join([char for char in text if char in symbols])
  return text
