"""ValueError: The features can't be aligned because the key _prompt of features
 {'_prompt': [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None), 'time': Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None)}],
 '_response': [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None), 'time': Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None)}],
  '_system': Value(dtype='string', id=None),
  '_tools': Value(dtype='string', id=None),
  '_images': Value(dtype='null', id=None),
  '_videos': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None)} has unexpected type -
  [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None), 'time': Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None)}]
  (expected either [{'content': Value(dtype='string', id=None), 'role': Value(dtype='string', id=None), 'time': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None)}]
  or Value("null")."""