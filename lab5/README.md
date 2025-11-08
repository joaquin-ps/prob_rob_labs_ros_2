To reproduce plots, a venv must be used to not conflict with ros2 python install.

Install venv
```
sudo apt install python3.10-venv
```

Make environemnt:
```
python3 -m venv .plotenv
source .plotenv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# python -m pip install "numpy<2" "matplotlib<3.9" pandas
```

Check our data: 
```
(.plotenv) python check_data.py 
```

Plot regressions:
```
(.plotenv) python fit_model_data.py 
```

