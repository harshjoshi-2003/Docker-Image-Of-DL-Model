FROM python:3.9

WORKDIR C:\Users\HARSH\Desktop\5th sem Project
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python3", "./mal_det_dl.py"]