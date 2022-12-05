Photometric-Stereo
==================

Run codes
----
Install packages:
```shell
pip install requirements.txt
```

Run lambert:
```shell
python main.py
```
Run improved lambert:
```text
python main.py --improve
```

Run with specific task e.g. "cat":
```text
python main.py --name cat
```

Note
----
You may need to change the `DATA_DIR` and `RESULT_DIR` in `main.py`, to customize input and output locations.