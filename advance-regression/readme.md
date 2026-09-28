mkdir data/raw, data/processed, notebooks, src, models
Move-Item train.csv, test.csv, sample_submission.csv, data_description.txt data\raw\
python -m venv venv
.\venv\Scripts\Activate.ps1
select kernel
membuat requirements
install requirements.txt
membuat setup.py
pip install -e .


Yang belum: 

* 1. Memahami bagian multikolinearitas ke atas, tapi saya mungkin tetap bisa menulis ke bagian atasnya tapi dengan lebih sederhana, kalau mau dimengerti nanti dulu.

* 2. Membuat di bagian README.md dan Modular file, tapi saya bingung ini dikerjakan sebelum atau sesudah deep learning.