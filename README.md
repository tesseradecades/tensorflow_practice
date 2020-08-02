# tensorflow_practice

## Setup

1.  Using [Python 3.8.5](https://www.python.org/downloads/release/python-385/)

2.  `python -m venv --system-site-packages .\venv`

3.  `source .venv/Scripts/activate

4.  `pip install -r requirements.txt`

5. `python <exercise file name>`

6. `deactivate`

You will often see the following
```
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

This is used to quiet Tensorflow's many warning and info messages in the console so that it is easier to focus on the concepts at hand.