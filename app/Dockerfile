FROM python:3.8
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

ENV SECRET_KEY='_5y2L"F4Q8z\n\xec]/'
ENV DATABASE_PATH='instance/database.db'

EXPOSE 5000
CMD ["python", "app.py"]