FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.bot.txt /app/requirements.bot.txt
COPY requirements.backtest.txt /app/requirements.backtest.txt
RUN pip install --no-cache-dir -r /app/requirements.backtest.txt

COPY . /app

CMD ["python", "main.py"]
