FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python scripts/normalise_theme_dict.py --in dicts/walmart_theme_dict_original.json --out dicts/theme_dict_core.json && \
    python scripts/merge_theme_patch.py --core dicts/theme_dict_core.json --patch dicts/theme_dict_patch.yaml --out dicts/theme_dict_core.json && \
    python scripts/build_matcher.py
CMD ["python", "scripts/run_pipeline.py", "--csv", "data/comments_raw.csv"]
