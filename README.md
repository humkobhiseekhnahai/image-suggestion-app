
# image-suggestion-app

Streamlit UI (Dark theme) for an academic image → suggestions model.

## Files
- `app.py` — Streamlit app
- `requirements.txt` — Python deps
- `model.pth` — trained weights
- `idx_to_suggestion.pkl` — label mapping (index → suggestion)
- `.streamlit/config.toml` — Dark theme

## Local run (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Create a **public GitHub repo** (e.g., `image-suggestion-app`).
2. **Upload all files** in this folder, including the `.streamlit/config.toml`.
   - If `model.pth` > 25MB, use **Git LFS**:
     ```bash
     git lfs install
     git lfs track "*.pth"
     git add .gitattributes model.pth
     git commit -m "Track model with LFS"
     git push
     ```
3. Go to https://share.streamlit.io → **New app** → choose the repo, branch `main`, and **main file** `app.py`.
4. Deploy. Streamlit will auto-install dependencies and launch the app in Dark mode.

## Notes
- Threshold for multi-label predictions is fixed at 0.5 in `app.py`. Adjust if needed.
- Inputs supported: `.jpg`, `.jpeg`, `.png`.
